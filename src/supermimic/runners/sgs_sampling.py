"""Success Guided Sampling over a fixed set of motion start bins."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SGSCfg:
  target: float = 0.66
  kappa: float = 5.0
  temperature: float = 2.0
  eps: float = 1e-8
  history: int = 100
  bin_frames: int = 50
  eval_steps: int = 50
  root_pos_threshold: float = 0.15
  body_pos_threshold: float = 0.15


class SGSSampler(nn.Module):
  def __init__(
    self,
    num_bins: int,
    valid: torch.Tensor,
    *,
    history: int = 100,
    target: float = 0.66,
    kappa: float = 5.0,
    temperature: float = 2.0,
    eps: float = 1e-8,
  ):
    super().__init__()
    if history < 1 or valid.shape != (num_bins,) or not valid.any():
      raise ValueError("SGS requires a positive history and at least one valid bin")
    if not (0 <= target <= 1) or not math.isfinite(kappa) or kappa < 0:
      raise ValueError("Invalid SGS target or concentration")
    if not math.isfinite(temperature) or not math.isfinite(eps) or eps <= 0:
      raise ValueError("SGS temperature and epsilon must be finite, epsilon positive")
    self.target, self.kappa = target, kappa
    self.temperature, self.eps = max(temperature, 1.0), eps
    self.register_buffer("valid", valid.bool().clone())
    self.register_buffer(
      "history", torch.zeros(num_bins, history, dtype=torch.bool, device=valid.device)
    )
    self.register_buffer(
      "count", torch.zeros(num_bins, dtype=torch.long, device=valid.device)
    )
    self.register_buffer(
      "ptr", torch.zeros(num_bins, dtype=torch.long, device=valid.device)
    )
    # Unvisited configurations all start at p=0, with equal nonzero probability.
    self.register_buffer("success_rate", torch.zeros(num_bins, device=valid.device))
    self.register_buffer(
      "total_updates", torch.zeros((), dtype=torch.long, device=valid.device)
    )
    self._probabilities = None

  def probabilities(self) -> torch.Tensor:
    if self._probabilities is None:
      # Float64 avoids loss of 1-p+eps near p=1 and kernel underflow.
      p = self.success_rate.double()
      log_w = self.kappa * (
        self.target * (p + self.eps).log()
        + (1 - self.target) * (1 - p + self.eps).log()
      )
      w = log_w.exp().clamp_min(self.eps)
      logits = (w + self.eps).log() / self.temperature
      self._probabilities = (
        logits.masked_fill(~self.valid, -torch.inf).softmax(0).float()
      )
    return self._probabilities

  @torch.no_grad()
  def update(self, bins: torch.Tensor, success: torch.Tensor) -> None:
    if bins.numel() == 0:
      return
    if bins.ndim != 1 or success.shape != bins.shape or success.dtype != torch.bool:
      raise ValueError("Expected one Boolean outcome per bin index")
    if not self.valid[bins].all():
      raise ValueError("Cannot update excluded SGS bins")
    order = bins.argsort(stable=True)
    b, outcomes = bins[order], success[order]
    unique, counts = b.unique_consecutive(return_counts=True)
    starts = counts.cumsum(0) - counts
    local = torch.arange(b.numel(), device=b.device) - starts.repeat_interleave(counts)
    capacity = self.history.shape[1]
    # Keep the last H outcomes when a batch alone overflows a bin's ring.
    keep = local >= (counts - capacity).repeat_interleave(counts)
    slots = (self.ptr[b] + local) % capacity
    self.history[b[keep], slots[keep]] = outcomes[keep]
    self.ptr[unique] = (self.ptr[unique] + counts) % capacity
    self.count[unique] = (self.count[unique] + counts).clamp_max(capacity)
    self.success_rate[unique] = self.history[unique].sum(1).float() / self.count[unique]
    self.total_updates += b.numel()
    self._probabilities = None

  def sample(self, n: int) -> torch.Tensor:
    return self.probabilities().multinomial(n, replacement=True)

  def load_state_dict(self, state_dict, strict=True, assign=False):
    result = super().load_state_dict(state_dict, strict=strict, assign=assign)
    self._probabilities = None
    return result
