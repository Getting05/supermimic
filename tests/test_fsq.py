"""FSQ forward values, gradient flow, task BC and deploy consistency."""

import pytest
import torch

from supermimic.models import MultiEncoderAE
from supermimic.quantization import LatentBottleneck
from supermimic.runners.exporter import _export_onnx, _SingleEncoderOnnx
from tests.conftest import make_actor_obs, make_actor_specs


def test_levels_and_straight_through():
  x = torch.linspace(-3, 3, 1000, requires_grad=True)
  z = LatentBottleneck(True)(x)
  assert torch.equal(z.unique(), torch.arange(-4, 5).float())
  z.sum().backward()
  assert torch.isfinite(x.grad).all() and (x.grad > 0).all()
  assert torch.equal(LatentBottleneck(False)(x), x.tanh())


@pytest.mark.parametrize("fsq", [False, True])
def test_export_slice_and_multi_encoder_gradients(fsq, tmp_path):
  actor = MultiEncoderAE(
    68,
    29,
    make_actor_specs(),
    {"motion": 0.5, "velocity": 0.5},
    40 if fsq else 32,
    (32,),
    1.0,
    2,
    fsq=fsq,
  )
  td = make_actor_obs(8, history_length=2)
  actor(td)
  actor.auxiliary_loss().sum().backward()
  assert any(
    p.grad is not None and p.grad.abs().sum() > 0
    for p in actor.encoders["velocity"].parameters()
  )
  assert all(p.grad is None for p in actor.encoders["motion"].parameters())
  actor.eval()
  actor.set_slot_mix({"motion": 1.0})
  deploy = _SingleEncoderOnnx(actor, "motion", [68])
  inputs = torch.cat((td["history"].flatten(1), td["goal_motion"]), -1)
  torch.testing.assert_close(deploy(inputs), actor(td))
  traced = torch.jit.trace(deploy, inputs)
  torch.testing.assert_close(traced(inputs), actor(td))
  from onnx.reference import ReferenceEvaluator

  path = tmp_path / "actor.onnx"
  _export_onnx(deploy, (inputs,), ["obs"], ["actions"], str(path))
  exported = ReferenceEvaluator(str(path)).run(None, {"obs": inputs.numpy()})[0]
  torch.testing.assert_close(
    torch.from_numpy(exported), actor(td), atol=2e-6, rtol=2e-5
  )
