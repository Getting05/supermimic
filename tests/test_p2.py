"""P2 asset, FK/IK and named motion mapping contracts."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from supermimic.envs.motion_library import MotionLibrary
from supermimic.tasks.env_cfgs import p2_env_cfg
from supermimic.tasks.p2 import P2_JOINT_NAMES, _p2_spec
from supermimic.utils.analytic_ik import AnalyticIK
from supermimic.utils.forward_kinematics import ForwardKinematics


@pytest.fixture(scope="module")
def p2():
  model = _p2_spec().compile()
  joints = [
    j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
  ]
  robot = SimpleNamespace(
    joint_names=[model.joint(j).name for j in joints],
    body_names=[model.body(b).name for b in range(1, model.nbody)],
    indexing=SimpleNamespace(
      body_ids=torch.arange(1, model.nbody),
      joint_q_adr=torch.tensor(model.jnt_qposadr[joints]),
    ),
  )
  rows = torch.tensor(
    [robot.body_names.index(n) for n in p2_env_cfg().commands["motion"].body_names]
  )
  return model, robot, rows


def test_task_assets_and_actuators(p2):
  model, robot, _ = p2
  assert tuple(robot.joint_names) == P2_JOINT_NAMES
  assert (model.nq, model.nv, model.nsensor) == (36, 35, 2)
  for maximal in (False, True):
    cfg = p2_env_cfg(maxcoord=maximal)
    assert cfg.decimation * cfg.sim.mujoco.timestep == 0.02
    assert len(cfg.actions["joint_pos"].scale) == 29
    assert "waist_roll_link" in cfg.commands["motion"].body_names
    assert "torso_link" not in cfg.commands["motion"].body_names
    assert ("body_pos_w" in cfg.observations["full_state"].terms) == maximal


def test_fk_and_maxcoord_ik_match_mujoco(p2):
  model, robot, rows = p2
  data = mujoco.MjData(model)
  rng = np.random.default_rng(2)
  data.qpos[:3] = [0.2, -0.1, 0.7]
  data.qpos[3:7] = [0.96, 0.1, -0.2, 0.15]
  data.qpos[3:7] /= np.linalg.norm(data.qpos[3:7])
  data.qpos[7:] = rng.uniform(-0.1, 0.1, 29)
  data.qvel[:] = rng.normal(size=35)
  mujoco.mj_forward(model, data)
  body_ids = robot.indexing.body_ids[rows].numpy()
  velocity = np.zeros((len(rows), 6))
  for i, b in enumerate(body_ids):
    mujoco.mj_objectVelocity(
      model, data, mujoco.mjtObj.mjOBJ_BODY, int(b), velocity[i], 0
    )
  # mj_objectVelocity's body output is at the inertial COM; shift to link origin.
  lin = velocity[:, 3:] + np.cross(
    velocity[:, :3], data.xpos[body_ids] - data.xipos[body_ids]
  )
  root_ang = data.cvel[1, :3]

  def t(x):
    return torch.tensor(x, dtype=torch.float32)[None]

  joints = t(data.qpos[7:]).requires_grad_()
  fk = ForwardKinematics(model, robot, rows, "cpu")
  pos, quat, vl, va = fk(
    t(data.qpos[:3]),
    t(data.qpos[3:7]),
    joints,
    t(data.qvel[:3]),
    t(root_ang),
    t(data.qvel[6:]),
  )
  torch.testing.assert_close(pos[0], t(data.xpos[body_ids])[0], atol=2e-6, rtol=2e-6)
  torch.testing.assert_close(vl[0], t(lin)[0], atol=3e-6, rtol=3e-6)
  torch.testing.assert_close(va[0], t(velocity[:, :3])[0], atol=3e-6, rtol=3e-6)
  ik = AnalyticIK(
    model,
    robot.indexing.body_ids,
    robot.indexing.joint_q_adr,
    rows,
    "cpu",
    torch.float32,
  )
  rotations = t(data.xmat[body_ids].reshape(-1, 3, 3))
  q, qd = ik(rotations, va)
  torch.testing.assert_close(q, joints, atol=3e-6, rtol=3e-6)
  torch.testing.assert_close(qd, t(data.qvel[6:]), atol=3e-6, rtol=3e-6)
  pos.square().sum().backward()
  assert torch.isfinite(joints.grad).all() and joints.grad.abs().sum() > 0


def write_static_motion(path, model, robot, frames=100):
  data = mujoco.MjData(model)
  data.qpos[2] = 0.645
  data.qpos[7:] = [
    p2_env_cfg().scene.entities["robot"].init_state.joint_pos[n]
    for n in robot.joint_names
  ]
  mujoco.mj_forward(model, data)

  def repeat(x):
    return np.repeat(x[None], frames, axis=0).astype(np.float32)

  np.savez(
    path,
    fps=np.array([50.0]),
    joint_names=robot.joint_names,
    body_names=robot.body_names,
    joint_pos=repeat(data.qpos[7:]),
    joint_vel=repeat(np.zeros(29)),
    body_pos_w=repeat(data.xpos[1:]),
    body_quat_w=repeat(data.xquat[1:]),
    body_lin_vel_w=repeat(np.zeros((model.nbody - 1, 3))),
    body_ang_vel_w=repeat(np.zeros((model.nbody - 1, 3))),
  )


def test_named_multiclip_motion(p2, tmp_path):
  model, robot, rows = p2
  paths = [tmp_path / "a.npz", tmp_path / "b.npz"]
  for path, frames in zip(paths, [100, 60], strict=True):
    write_static_motion(path, model, robot, frames)
  motion = MotionLibrary([str(p) for p in paths], robot, rows, "cpu", 0.02)
  assert motion.motion_ends.tolist() == [100, 160]
  assert motion.body_pos_w.shape == (160, 14, 3)
  assert motion.root_lin_vel_w.shape == (160, 3)


def test_convert_retarget_by_joint_name(p2, tmp_path):
  from supermimic.scripts.convert_p2_motion import convert_p2_retarget_to_npz

  model, robot, rows = p2
  source, output = tmp_path / "retarget.npz", tmp_path / "tracking.npz"
  angles = np.tile(np.linspace(-0.1, 0.1, 29), (100, 1))
  np.savez(
    source,
    fps=50.0,
    joint_names=robot.joint_names[::-1],
    joint_angles=angles[:, ::-1],
    base_frame_pos=np.tile([0.0, 0.0, 0.645], (100, 1)),
    base_frame_wxyz=np.tile([1.0, 0.0, 0.0, 0.0], (100, 1)),
  )
  convert_p2_retarget_to_npz(source, output)
  motion = MotionLibrary([str(output)], robot, rows, "cpu", 0.02)
  torch.testing.assert_close(
    motion.joint_pos, torch.tensor(angles, dtype=torch.float32)
  )
  assert motion.body_pos_w.shape == (100, 14, 3)
