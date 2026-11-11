"""SGS distribution, ring updates, horizon bookkeeping and persistence."""

from types import SimpleNamespace

import torch

from supermimic.envs.motion_command import TrackingMotionCommand
from supermimic.runners.sgs_sampling import SGSCfg, SGSSampler


def test_distribution_prefers_target_and_excludes_invalid():
  sampler = SGSSampler(4, torch.tensor([True, True, True, False]))
  torch.testing.assert_close(
    sampler.probabilities(), torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0])
  )
  bins = torch.arange(3).repeat_interleave(100)
  success = torch.cat(
    (torch.zeros(100), torch.arange(100) < 66, torch.ones(100))
  ).bool()
  sampler.update(bins, success)
  p = sampler.probabilities()
  assert p.argmax() == 1 and p[3] == 0
  restored = SGSSampler(4, sampler.valid)
  restored.load_state_dict(sampler.state_dict())
  torch.testing.assert_close(restored.probabilities(), p)


def test_ring_retains_latest_simultaneous_outcomes():
  sampler = SGSSampler(1, torch.ones(1, dtype=torch.bool), history=3)
  sampler.update(
    torch.zeros(5, dtype=torch.long), torch.tensor([False, False, True, True, True])
  )
  assert sampler.success_rate.item() == 1
  sampler.update(torch.tensor([0]), torch.tensor([False]))
  torch.testing.assert_close(sampler.success_rate, torch.tensor([2 / 3]))


def test_success_requires_full_horizon_and_failure_wins_timeout():
  cmd = object.__new__(TrackingMotionCommand)
  cmd.cfg = SimpleNamespace(sgs=SGSCfg())
  cmd._env = SimpleNamespace(
    common_step_counter=50,
    termination_manager=SimpleNamespace(
      terminated=torch.tensor([False, True, False, False]),
      time_outs=torch.tensor([True, True, True, False]),
    ),
  )
  cmd.sgs_sampler = SGSSampler(4, torch.ones(4, dtype=torch.bool))
  cmd._sgs_last_step = 50
  cmd._sgs_active = torch.ones(4, dtype=torch.bool)
  cmd._sgs_steps = torch.tensor([50, 50, 49, 25])
  cmd._sgs_bins = torch.arange(4)
  cmd._sgs_root = torch.zeros(4)
  cmd._sgs_body = torch.zeros(4)
  cmd._sgs_finished = torch.zeros(4)
  cmd._finalize_sgs(torch.arange(4))
  assert cmd.sgs_sampler.count.tolist() == [1, 1, 1, 0]
  assert cmd.sgs_sampler.success_rate.tolist() == [1, 0, 0, 0]
  assert not cmd._sgs_active.any()
