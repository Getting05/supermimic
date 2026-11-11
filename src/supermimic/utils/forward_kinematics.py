"""Differentiable FK from the actual scene's robot tree (P2 or G1)."""

import mujoco
import torch
from mjlab.utils.lab_api.math import quat_apply, quat_mul


class ForwardKinematics:
  def __init__(self, model, robot, tracked_rows, device):
    body_ids = robot.indexing.body_ids.cpu().tolist()
    row_of = {b: i for i, b in enumerate(body_ids)}
    col_of = {int(a): i for i, a in enumerate(robot.indexing.joint_q_adr)}
    self.rows = tracked_rows
    self.links = []
    for row, body in enumerate(body_ids):
      if row == 0:
        continue
      parent = row_of[int(model.body_parentid[body])]
      joint = int(model.body_jntadr[body])
      col = None
      axis, anchor, zero = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0
      if model.body_jntnum[body]:
        if (
          model.body_jntnum[body] != 1
          or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE
        ):
          raise ValueError("FK requires one free root and fixed/hinge child links")
        adr = int(model.jnt_qposadr[joint])
        col = col_of[adr]
        axis, anchor, zero = (
          model.jnt_axis[joint],
          model.jnt_pos[joint],
          model.qpos0[adr],
        )
      tensors = [
        torch.tensor(x, dtype=torch.float32, device=device)
        for x in (model.body_pos[body], model.body_quat[body], axis, anchor)
      ]
      self.links.append((parent, col, float(zero), *tensors))

  def __call__(
    self, root_pos, root_quat, joint_pos, root_lin_vel, root_ang_vel, joint_vel
  ):
    pos, quat = [root_pos], [root_quat]
    lin, ang = [root_lin_vel], [root_ang_vel]
    batch = root_pos.shape[:-1]
    for parent, col, zero, offset, rotation, axis, anchor in self.links:
      p = pos[parent] + quat_apply(quat[parent], offset.expand(*batch, 3))
      q = quat_mul(quat[parent], rotation.expand(*batch, 4))
      wj = torch.zeros_like(root_ang_vel)
      a = p
      if col is not None:
        a = p + quat_apply(q, anchor.expand(*batch, 3))
        world_axis = quat_apply(q, axis.expand(*batch, 3))
        half = (joint_pos[..., col : col + 1] - zero) * 0.5
        dq = torch.cat((half.cos(), half.sin() * axis), dim=-1)
        q = quat_mul(q, dq)
        p = a - quat_apply(q, anchor.expand(*batch, 3))
        wj = world_axis * joint_vel[..., col : col + 1]
      pos.append(p)
      quat.append(q)
      ang.append(ang[parent] + wj)
      lin.append(
        lin[parent]
        + torch.linalg.cross(ang[parent], p - pos[parent])
        + torch.linalg.cross(wj, p - a)
      )
    return tuple(
      torch.stack(values, -2)[..., self.rows, :] for values in (pos, quat, lin, ang)
    )
