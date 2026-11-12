"""Optional FSQ bottleneck shared by training and deployment."""

import torch
from torch import nn


class LatentBottleneck(nn.Module):
  """Continuous tanh, or the reproduced 9-level FSQ with straight-through round."""

  def __init__(self, fsq: bool = False):
    super().__init__()
    self.fsq = fsq

  def forward(self, value: torch.Tensor) -> torch.Tensor:
    latent = torch.tanh(value)
    if self.fsq:
      bounded = latent.float() * (4.5 * (1 - 1e-4))
      latent = bounded + (bounded.round() - bounded).detach()
    return latent
