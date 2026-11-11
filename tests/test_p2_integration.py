"""Opt-in CPU simulation + differentiable training smoke, no W&B or GPU needed.

Run SUPERMIMIC_SIM_TEST=1 pytest tests/test_p2_integration.py -q -s.
"""

import os

import pytest
import torch

from supermimic.envs.motion_command import configure_motion_command
from supermimic.runners.sgs_sampling import SGSCfg
from supermimic.runners.supermimic_runner import SuperMimicRunner, SuperMimicRunnerCfg
from supermimic.scripts.train_utils import create_env
from supermimic.tasks.env_cfgs import p2_env_cfg
from tests.test_p2 import p2, write_static_motion  # noqa: F401

pytestmark = pytest.mark.skipif(
  os.environ.get("SUPERMIMIC_SIM_TEST") != "1", reason="opt-in simulator smoke"
)


@pytest.mark.parametrize("maximal", [False, True])
def test_p2_training_and_sgs_horizon(p2, tmp_path, maximal):  # noqa: F811
  model, robot, _ = p2
  path = tmp_path / "stand.npz"
  write_static_motion(path, model, robot, 120)
  cfg = p2_env_cfg(maxcoord=maximal)
  cfg.commands["motion"].motion_file = str(path)
  configure_motion_command(cfg, sgs=SGSCfg())
  cfg.commands["motion"].pose_range = {}
  cfg.commands["motion"].velocity_range = {}
  cfg.commands["motion"].joint_position_range = (0.0, 0.0)
  # Exercise the full 50-step timeout even if an untrained policy falls.
  cfg.terminations["anchor_pos"].params["threshold"] = 100.0
  cfg.terminations["anchor_ori"].params["threshold"] = 100.0
  env = create_env(cfg, 2, "cpu")
  try:
    rcfg = SuperMimicRunnerCfg(
      representation="maxcoord" if maximal else "gencoord",
      actor_obs="proprio",
      history_length=2,
      fsq=True,
      sampling_mode="sgs",
      buffer_size=64,
      dynamics_hidden_dims=(32, 32),
      actor_hidden_dims=(32,),
      critic_hidden_dims=(32,),
    )
    rcfg.wm.batch_size = rcfg.policy.batch_size = 2
    rcfg.wm.horizon = rcfg.policy.horizon = 2
    rcfg.wm.num_batches = rcfg.policy.num_batches = 1
    runner = SuperMimicRunner(env, rcfg, "P2-smoke", device="cpu")
    # The smoke exercises eager CPU updates; compilation has its own torch tests.
    runner.world_model_learner._compiled_rollout = runner.world_model_learner._rollout
    runner.world_model_learner._compiled_compute_losses = (
      runner.world_model_learner._compute_losses
    )
    runner.world_model_learner._compiled_represent = runner._representation.represent
    runner._seed_sim_history()
    cmd = runner.motion_cmd
    initial = cmd.time_steps.clone()
    for i in range(49):
      runner.step()
      assert torch.equal(cmd.time_steps, initial + i + 1)
    assert cmd.sgs_sampler.total_updates == 0
    runner.step()
    assert cmd.sgs_sampler.total_updates == 2
    assert cmd._sgs_steps.tolist() == [0, 0]
    assert (cmd.time_steps <= 70).all()
    batch = runner.replay_buffer.sample(2, 2)
    wm_logs = runner.world_model_learner.update([batch])
    policy_logs = runner.policy_learner.update()
    assert all(
      torch.isfinite(torch.as_tensor(v)).all() for v in (wm_logs | policy_logs).values()
    )
    state = cmd.sgs_state_dict()
    cmd.load_sgs_state_dict(state)
    assert cmd.sgs_sampler.total_updates == 2
    cmd.reset_to_frame(torch.arange(2), 0)
    assert not cmd._sgs_active.any()
  finally:
    env.close()
