"""Public-mjlab motion adapter, with optional success-guided RSI."""

from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path

import torch
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg

from supermimic.envs.motion_library import MotionLibrary, motion_paths
from supermimic.runners.sgs_sampling import SGSCfg, SGSSampler
from supermimic.utils.forward_kinematics import ForwardKinematics


class TrackingMotionCommand(MotionCommand):
  def __init__(self, cfg, env):
    paths = motion_paths(cfg.motion_file)
    # Public mjlab loads one file in its constructor. Replace the loader with
    # the named, multi-clip version without changing the dependency globally.
    super().__init__(replace(cfg, motion_file=paths[0]), env)
    self.cfg = cfg
    self.motion = MotionLibrary(
      paths, self.robot, self.body_indexes, self.device, env.step_dt
    )
    self.motion.fk = ForwardKinematics(
      env.sim.mj_model, self.robot, self.body_indexes, self.device
    )
    self.bin_count = int(self.motion.time_step_total * env.step_dt) + 1
    for name in ("bin_failed_count", "_current_bin_failed"):
      setattr(self, name, torch.zeros(self.bin_count, device=self.device))
    self._external_sampling_weights = torch.ones(self.bin_count, device=self.device)
    self._just_reset = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
    self.sgs_sampler = None
    if cfg.sgs_enabled:
      s = cfg.sgs
      if (
        s.bin_frames < 1
        or s.eval_steps < 1
        or min(s.root_pos_threshold, s.body_pos_threshold) <= 0
      ):
        raise ValueError("SGS bin size, horizon and error thresholds must be positive")
      starts, ends = [], []
      begin = 0
      for end in self.motion.motion_ends.tolist():
        for start in range(begin, end, s.bin_frames):
          starts.append(start)
          ends.append(min(start + s.bin_frames, end - s.eval_steps + 1))
        begin = end
      self._sgs_starts = torch.tensor(starts, device=self.device)
      self._sgs_ends = torch.tensor(ends, device=self.device)
      self.sgs_sampler = SGSSampler(
        len(starts),
        self._sgs_ends > self._sgs_starts,
        history=s.history,
        target=s.target,
        kappa=s.kappa,
        temperature=s.temperature,
        eps=s.eps,
      )
      self._sgs_active = torch.zeros(
        self.num_envs, dtype=torch.bool, device=self.device
      )
      self._sgs_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
      self._sgs_bins = torch.zeros_like(self._sgs_steps)
      self._sgs_root = torch.zeros(self.num_envs, device=self.device)
      self._sgs_body = torch.zeros_like(self._sgs_root)
      self._sgs_last_step = -1
      self._sgs_finished = torch.zeros(4, device=self.device)

  def _external_sampling(self, env_ids):
    if self.sgs_sampler is None:
      bins = self._external_sampling_weights.multinomial(len(env_ids), replacement=True)
      # Same integer frame->bin partition as ReturnSampler._bins.
      total = self.motion.time_step_total
      low = (bins * total + self.bin_count - 1) // self.bin_count
      high = ((bins + 1) * total + self.bin_count - 1) // self.bin_count
    else:
      bins = self.sgs_sampler.sample(len(env_ids))
      self._sgs_bins[env_ids] = bins
      low, high = self._sgs_starts[bins], self._sgs_ends[bins]
    self.time_steps[env_ids] = (
      low + (torch.rand(len(env_ids), device=self.device) * (high - low)).long()
    )

  def _resample_command(self, env_ids):
    if self.sgs_sampler is not None:
      self._finalize_sgs(env_ids)
    super()._resample_command(env_ids)
    self._just_reset[env_ids] = True
    if self.sgs_sampler is not None:
      self._sgs_active[env_ids] = True

  def _update_command(self):
    # Reset calls must expose the sampled frame, including env.reset(dt=0).
    clip = torch.searchsorted(self.motion.motion_ends, self.time_steps, right=True)
    last = self.motion.motion_ends[clip] - 1
    self.time_steps.copy_(
      torch.minimum(self.time_steps + (~self._just_reset).long(), last)
    )
    self._just_reset.zero_()
    self.update_relative_body_poses()
    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count.lerp_(self._current_bin_failed, self.cfg.adaptive_alpha)
      self._current_bin_failed.zero_()

  def reset_to_frame(self, env_ids, frame):
    super().reset_to_frame(env_ids, frame)
    self._just_reset[env_ids] = True
    if self.sgs_sampler is not None:
      self._sgs_active[env_ids] = False

  def record_sgs_step(self):
    if self._sgs_last_step != self._env.common_step_counter:
      self._sgs_root += (
        torch.linalg.vector_norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        * self._sgs_active
      )
      self._sgs_body += (
        torch.linalg.vector_norm(self.body_pos_w - self.robot_body_pos_w, dim=-1).mean(
          -1
        )
        * self._sgs_active
      )
      self._sgs_steps += self._sgs_active.long()
      self._sgs_last_step = self._env.common_step_counter
    return self._sgs_active & (self._sgs_steps >= self.cfg.sgs.eval_steps)

  def _finalize_sgs(self, env_ids):
    manager = self._env.termination_manager
    ended = manager.terminated[env_ids] | manager.time_outs[env_ids]
    # A manual reset outside the terminating step discards an unfinished episode.
    ended &= self._sgs_last_step == self._env.common_step_counter
    ids = env_ids[self._sgs_active[env_ids] & ended & (self._sgs_steps[env_ids] > 0)]
    root = self._sgs_root[ids] / self._sgs_steps[ids]
    body = self._sgs_body[ids] / self._sgs_steps[ids]
    success = (
      (self._sgs_steps[ids] == self.cfg.sgs.eval_steps)
      & ~manager.terminated[ids]
      & (root < self.cfg.sgs.root_pos_threshold)
      & (body < self.cfg.sgs.body_pos_threshold)
    )
    self.sgs_sampler.update(self._sgs_bins[ids], success)
    self._sgs_finished += torch.stack(
      (torch.ones_like(root).sum(), success.sum(), root.sum(), body.sum())
    )
    self._sgs_active[env_ids] = False
    for value in (self._sgs_steps, self._sgs_root, self._sgs_body):
      value[env_ids] = 0

  def sgs_state_dict(self):
    return {
      "config": asdict(self.cfg.sgs),
      "clips": [
        (Path(p).name, int(n))
        for p, n in zip(
          self.motion.paths, self.motion.motion_ends.tolist(), strict=True
        )
      ],
      "sampler": self.sgs_sampler.state_dict(),
    }

  def load_sgs_state_dict(self, state):
    expected = self.sgs_state_dict()
    if state["config"] != expected["config"] or state["clips"] != expected["clips"]:
      raise ValueError(
        "SGS state requires the same ordered clips and sampler configuration"
      )
    self.sgs_sampler.load_state_dict(state["sampler"])

  def sgs_logs(self):
    sampler = self.sgs_sampler
    p = sampler.probabilities()[sampler.valid]
    rates = sampler.success_rate[sampler.valid]
    count, successes, root, body = self._sgs_finished
    logs = {
      "sgs/success_rate_mean": rates.mean(),
      "sgs/sampling_entropy": -(p * p.log()).sum(),
      "sgs/valid_bins": sampler.valid.sum(),
      "sgs/excluded_bins": (~sampler.valid).sum(),
      "sgs/unvisited_bins": (sampler.count[sampler.valid] == 0).sum(),
      "sgs/episode_success_rate": successes / count.clamp_min(1),
      "sgs/root_error": root / count.clamp_min(1),
      "sgs/body_error": body / count.clamp_min(1),
      "sgs/total_outcomes": sampler.total_updates.clone(),
    }
    self._sgs_finished.zero_()
    return {k: v.item() for k, v in logs.items()}


@dataclass(kw_only=True)
class TrackingMotionCommandCfg(MotionCommandCfg):
  sgs_enabled: bool = False
  sgs: SGSCfg = field(default_factory=SGSCfg)

  def build(self, env):
    return TrackingMotionCommand(self, env)


def motion_end(env, command_name="motion"):
  cmd = env.command_manager.get_term(command_name)
  clip = torch.searchsorted(cmd.motion.motion_ends, cmd.time_steps, right=True)
  return cmd.time_steps >= cmd.motion.motion_ends[clip] - 1


def sgs_horizon(env, command_name="motion"):
  return env.command_manager.get_term(command_name).record_sgs_step()


def configure_motion_command(env_cfg, *, sgs=None):
  current = env_cfg.commands["motion"]
  values = {f.name: getattr(current, f.name) for f in fields(MotionCommandCfg)}
  cfg = TrackingMotionCommandCfg(**values)
  if sgs is not None:
    if (
      env_cfg.episode_length_s
      < sgs.eval_steps * env_cfg.sim.mujoco.timestep * env_cfg.decimation
    ):
      raise ValueError("Episode timeout is shorter than the SGS evaluation horizon")
    cfg.sgs_enabled, cfg.sgs, cfg.sampling_mode = True, sgs, "external"
    env_cfg.terminations["sgs_horizon"] = TerminationTermCfg(
      func=sgs_horizon, time_out=True
    )
  env_cfg.commands["motion"] = cfg
  env_cfg.terminations["motion_end"] = TerminationTermCfg(
    func=motion_end, time_out=True
  )
