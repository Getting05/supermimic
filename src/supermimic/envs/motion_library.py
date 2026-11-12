"""Named NPZ motion loading for the public mjlab adapter."""

import glob
from pathlib import Path

import numpy as np
import torch


def motion_paths(patterns):
  patterns = [patterns] if isinstance(patterns, str) else patterns
  paths = [
    p
    for pattern in patterns
    for p in sorted(glob.glob(str(Path(pattern).expanduser())))
  ]
  if not paths:
    raise FileNotFoundError(f"No motion NPZ files match {patterns}")
  return paths


class MotionLibrary:
  def __init__(self, paths, robot, tracked_rows, device, step_dt):
    arrays = {
      k: []
      for k in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
      )
    }
    lengths = []
    for path in paths:
      with np.load(path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isclose(fps * step_dt, 1.0):
          raise ValueError(f"{path}: motion fps {fps} must match control dt {step_dt}")
        joint_order = list(range(len(robot.joint_names)))
        body_order = list(range(len(robot.body_names)))
        for key, expected, order in (
          ("joint_names", robot.joint_names, joint_order),
          ("body_names", robot.body_names, body_order),
        ):
          if key in data:
            names = data[key].tolist()
            order[:] = [names.index(n) for n in expected]
        length = len(data["joint_pos"])
        if length < 2:
          raise ValueError(f"{path}: motion needs at least two frames")
        lengths.append(length)
        for key in arrays:
          value = data[key][:, joint_order if key.startswith("joint") else body_order]
          if len(value) != length or not np.isfinite(value).all():
            raise ValueError(f"{path}: invalid {key}")
          arrays[key].append(
            torch.as_tensor(value.copy(), device=device, dtype=torch.float32)
          )
    self.paths = paths
    self.motion_ends = torch.tensor(lengths, device=device).cumsum(0)
    self.time_step_total = sum(lengths)
    for key, values in arrays.items():
      value = torch.cat(values)
      if key.startswith("body"):
        setattr(self, "_" + key, value)
        setattr(self, key, value[:, tracked_rows])
      else:
        setattr(self, key, value)
    for name in ("pos", "quat", "lin_vel", "ang_vel"):
      setattr(self, "root_" + name + "_w", getattr(self, "_body_" + name + "_w")[:, 0])

  def body_state(self, frames):
    return tuple(
      getattr(self, "body_" + name + "_w")[frames]
      for name in ("pos", "quat", "lin_vel", "ang_vel")
    )
