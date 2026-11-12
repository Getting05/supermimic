"""Convert named PyRoki P2 retarget NPZ files to 50 Hz tracking NPZ.

Resampling/FK conventions ported from Getting05/SuperMimic-Reproduced.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import tyro
from scipy.spatial.transform import Rotation


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
  return q[..., [3, 0, 1, 2]]


def quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
  return q[..., [1, 2, 3, 0]]


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
  w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
  return np.stack(
    [
      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
      w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ],
    axis=-1,
  )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
  return q * np.array([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def rotvec_from_quat(q: np.ndarray) -> np.ndarray:
  """Rotation vector (angle * axis) of wxyz quaternions, numerically stable near identity."""
  q = q / np.linalg.norm(q, axis=-1, keepdims=True)
  q = np.where(q[..., :1] < 0, -q, q)
  v = q[..., 1:]
  s = np.linalg.norm(v, axis=-1)
  angle = 2.0 * np.arctan2(s, q[..., 0])
  scale = np.where(s > 1e-9, angle / np.where(s > 1e-9, s, 1.0), 2.0)
  return v * scale[..., None]


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate world vectors into the body frame: R(q)^T v."""
  r = Rotation.from_quat(quat_wxyz_to_xyzw(q))
  return r.inv().apply(v)


@dataclass
class ResampledMotion:
  fps: float
  root_pos: np.ndarray  # (T,3)
  root_quat: np.ndarray  # (T,4) wxyz
  joint_pos: np.ndarray  # (T,29)
  root_lin_vel_w: np.ndarray  # (T,3) world
  root_ang_vel_w: np.ndarray  # (T,3) world
  joint_vel: np.ndarray  # (T,29)

  @property
  def num_frames(self) -> int:
    return int(self.root_pos.shape[0])


def resample_motion(
  raw: np.ndarray, input_fps: float, output_fps: float
) -> ResampledMotion:
  """Resample a raw CSV array to ``output_fps`` and compute velocities (mjlab conventions)."""
  input_dt = 1.0 / input_fps
  output_dt = 1.0 / output_fps
  n_in = raw.shape[0]
  duration = (n_in - 1) * input_dt
  # Include the end point when it lands on the output grid (same tolerance as mjlab).
  n_out = math.floor(duration / output_dt + 1e-6) + 1
  times = np.arange(n_out, dtype=np.float64) * output_dt
  phase = times / duration if duration > 0 else np.zeros_like(times)
  idx0 = np.floor(phase * (n_in - 1)).astype(np.int64)
  idx0 = np.clip(idx0, 0, n_in - 1)
  idx1 = np.minimum(idx0 + 1, n_in - 1)
  blend = (phase * (n_in - 1) - idx0)[:, None]

  pos_in = raw[:, 0:3]
  quat_in = quat_xyzw_to_wxyz(raw[:, 3:7])
  quat_in = quat_in / np.linalg.norm(quat_in, axis=-1, keepdims=True)
  dof_in = raw[:, 7:]

  root_pos = pos_in[idx0] * (1 - blend) + pos_in[idx1] * blend
  joint_pos = dof_in[idx0] * (1 - blend) + dof_in[idx1] * blend

  # Vectorised slerp between the two bracketing input quaternions (shortest path).
  q0, q1 = quat_in[idx0], quat_in[idx1]
  dot = np.sum(q0 * q1, axis=-1, keepdims=True)
  q1 = np.where(dot < 0, -q1, q1)
  dot = np.abs(dot)
  theta = np.arccos(np.clip(dot, -1.0, 1.0))
  sin_theta = np.sin(theta)
  lin = dot > 0.9995
  w0 = np.where(
    lin, 1 - blend, np.sin((1 - blend) * theta) / np.where(lin, 1.0, sin_theta)
  )
  w1 = np.where(lin, blend, np.sin(blend * theta) / np.where(lin, 1.0, sin_theta))
  root_quat = w0 * q0 + w1 * q1
  root_quat = root_quat / np.linalg.norm(root_quat, axis=-1, keepdims=True)

  # Velocities.
  if n_out >= 2:
    root_lin_vel = np.gradient(root_pos, output_dt, axis=0)
    joint_vel = np.gradient(joint_pos, output_dt, axis=0)
  else:
    root_lin_vel = np.zeros_like(root_pos)
    joint_vel = np.zeros_like(joint_pos)
  root_ang_vel = _so3_derivative(root_quat, output_dt)

  return ResampledMotion(
    fps=output_fps,
    root_pos=root_pos,
    root_quat=root_quat,
    joint_pos=joint_pos,
    root_lin_vel_w=root_lin_vel,
    root_ang_vel_w=root_ang_vel,
    joint_vel=joint_vel,
  )


def _so3_derivative(q: np.ndarray, dt: float) -> np.ndarray:
  """World-frame angular velocity by central differences; first/last sample repeated."""
  n = q.shape[0]
  if n < 3:
    return np.zeros((n, 3), dtype=q.dtype)
  q_prev, q_next = q[:-2], q[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = rotvec_from_quat(q_rel) / (2.0 * dt)
  return np.concatenate([omega[:1], omega, omega[-1:]], axis=0)


@dataclass
class BodyStates:
  body_names: list[str]
  pos_w: np.ndarray  # (T, nb, 3) link frame origin
  quat_w: np.ndarray  # (T, nb, 4) wxyz
  lin_vel_w: np.ndarray  # (T, nb, 3) link frame origin velocity, world
  ang_vel_w: np.ndarray  # (T, nb, 3) world


def forward_kinematics(model: mujoco.MjModel, motion: ResampledMotion) -> BodyStates:
  """Body link poses and velocities for every frame via CPU MuJoCo (no contacts)."""
  data = mujoco.MjData(model)
  T = motion.num_frames
  nb = model.nbody - 1  # exclude world
  pos_w = np.empty((T, nb, 3))
  quat_w = np.empty((T, nb, 4))
  lin_w = np.empty((T, nb, 3))
  ang_w = np.empty((T, nb, 3))
  root_id = model.jnt_bodyid[0]

  # Free-joint qvel: world linear velocity, **body-frame** angular velocity.
  ang_vel_b = quat_rotate_inverse(motion.root_quat, motion.root_ang_vel_w)

  qpos = np.zeros(model.nq)
  qvel = np.zeros(model.nv)
  for t in range(T):
    qpos[0:3] = motion.root_pos[t]
    qpos[3:7] = motion.root_quat[t]
    qpos[7:] = motion.joint_pos[t]
    qvel[0:3] = motion.root_lin_vel_w[t]
    qvel[3:6] = ang_vel_b[t]
    qvel[6:] = motion.joint_vel[t]
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    pos_w[t] = data.xpos[1:]
    quat_w[t] = data.xquat[1:]
    cvel = data.cvel[1:]
    offset = data.subtree_com[root_id][None, :] - data.xpos[1:]
    lin_w[t] = cvel[:, 3:6] - np.cross(cvel[:, 0:3], offset)
    ang_w[t] = cvel[:, 0:3]

  names = [
    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody)
  ]
  return BodyStates(names, pos_w, quat_w, lin_w, ang_w)


def convert_p2_retarget_to_npz(source: Path, out_path: Path) -> None:
  """Convert a ProtoMotions PyRoki P2 clip, matching joints by name."""
  from supermimic.tasks.p2 import P2_JOINT_NAMES, load_p2_model

  with np.load(source) as data:
    names = list(data["joint_names"])
    if len(set(names)) != len(names):
      raise ValueError(f"{source}: duplicate joint names")
    order = [names.index(name) for name in P2_JOINT_NAMES]
    fps = float(data["fps"].item())
    if not math.isfinite(fps) or fps <= 0:
      raise ValueError(f"{source}: expected positive source fps")
    raw = np.concatenate(
      [
        data["base_frame_pos"],
        data["base_frame_wxyz"][:, [1, 2, 3, 0]],
        data["joint_angles"][:, order],
      ],
      axis=1,
    )
  motion = resample_motion(raw, fps, 50.0)
  _write_motion_npz(
    out_path, motion, load_p2_model(), P2_JOINT_NAMES, source, fps, 50.0
  )


def _write_motion_npz(
  out_path, motion, model, joint_names, source, input_fps, output_fps
):
  bodies = forward_kinematics(model, motion)
  out_path = Path(out_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  np.savez(
    out_path,
    fps=np.array([output_fps], dtype=np.float32),
    joint_pos=motion.joint_pos.astype(np.float32),
    joint_vel=motion.joint_vel.astype(np.float32),
    body_pos_w=bodies.pos_w.astype(np.float32),
    body_quat_w=bodies.quat_w.astype(np.float32),
    body_lin_vel_w=bodies.lin_vel_w.astype(np.float32),
    body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
    joint_names=np.array(joint_names),
    body_names=np.array(bodies.body_names),
    source=np.array(str(source)),
    input_fps=np.array([input_fps], dtype=np.float32),
  )


def main(source: Path, output: Path) -> None:
  """Convert one retarget NPZ, or all *_retargeted.npz under a directory."""
  if source.is_file():
    convert_p2_retarget_to_npz(source, output)
  else:
    files = sorted(source.rglob("*_retargeted.npz"))
    if not files:
      raise FileNotFoundError(f"No retargeted motions under {source}")
    for path in files:
      convert_p2_retarget_to_npz(path, output / path.relative_to(source))


if __name__ == "__main__":
  tyro.cli(main)
