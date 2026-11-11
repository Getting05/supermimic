# P2, optional FSQ and optional SGS

Both `SuperMimic-GenCoord-P2` and `SuperMimic-MaxCoord-P2` use the actual
docker150 `astro_p2_supermimic.xml`, its meshes, 29 controlled joints and the
standard position-PD configuration from SuperMimic-Reproduced. Physics runs at
200 Hz and policy actions at 50 Hz. The head is fixed. Advanced delayed/motor-envelope
actuators are not enabled. Root tracking is at `pelvis`; torso tracking/COM
randomization uses `waist_roll_link`.

P2 training defaults to sensor proprioception (93 values) with ten history frames.
The existing target-repository motion goal, world model, rewards and learner are
retained. This is feature integration, not checkpoint compatibility with the source
repository's 730-dimensional goal or its complete training recipe.

## Motion data

Use P2 tracking NPZs, individually, as a glob, or as a comma-separated list. The
loader uses `joint_names` and `body_names` when present to map into the actual
robot's order. Files must have `fps=50`, joint positions/velocities, and body link
positions, wxyz quaternions, linear/angular velocities in world coordinates.
Clip boundaries remain separate in the concatenated library. Packed `.pt`
MotionLibrary files from SuperMimic-Reproduced are not accepted directly.

Convert PyRoki P2 retarget NPZs (named joints and `base_frame_wxyz`) with:

```bash
uv run python -m supermimic.scripts.convert_p2_motion \
  --source /data/p2_retarget --output /data/p2_tracking
```

This reorders joints by name, resamples to 50 Hz and computes body data with the
bundled P2 model. It preserves subdirectories when converting a directory.

## Independent switches

Baseline P2 with continuous latent and return sampling:

```bash
uv run python -m supermimic.scripts.train SuperMimic-GenCoord-P2 \
  --motion-file '/data/p2_tracking/*.npz' \
  --runner.run-name p2_continuous_return
```

Enable FSQ and SGS, with either representation:

```bash
uv run python -m supermimic.scripts.train SuperMimic-MaxCoord-P2 \
  --motion-file '/data/p2_tracking/*.npz' \
  --runner.fsq --runner.sampling-mode sgs \
  --runner.run-name p2_maxcoord_fsq_sgs
```

`--runner.fsq` defaults to false. When enabled, all task encoders use a shared
40-dimensional latent contract, each coordinate quantized to nine integer levels
(-4 through 4): `tanh(x) * 4.5 * (1 - 1e-4)`, then straight-through rounding.
`runner.latent_dim` applies only to the continuous mode (default 32). The same
bottleneck is used in the actor, latent BC and all single-task ONNX exports.
Continuous and FSQ checkpoints have different shapes; select the matching config.

`--runner.sampling-mode` accepts `return` (default), `uniform`, `adaptive`, `start`
and `sgs`. Both FSQ and SGS can also be enabled for G1. Switching only the sampling
mode does not change model shapes or replay sampling. For a controlled comparison,
hold `runner.actor_obs`, `history_length`, datasets and training settings fixed.

## SGS definition

SGS uses per-clip 50-frame bins and a rolling history of 100 Boolean outcomes.
The beta kernel uses target 0.66, concentration 5, temperature 2 and epsilon 1e-8,
matching the source implementation. Unvisited valid bins start uniformly.
Each sampled start must leave 50 frames, including the start, in the same clip.
The reset call does not advance the reference; action 50 is checked against
start + 49. Invalid tail bins are excluded, and the final valid bin is truncated
before sampling rather than clamping many samples onto one frame.

An episode succeeds after exactly 50 actions without failure termination, when
both the mean root error and the mean tracked-body error are less than 0.15 m.
Errors are world-space link-origin distances, without root alignment. Failure
wins over a simultaneous timeout. Only the episode's starting bin is updated.
Manual resets discard unfinished episodes; `reset_to_frame` evaluation does not
train the sampler. Error accumulation and outcome recording happen before reset.

All parameters are exposed under `--runner.sgs.*`, e.g.
`--runner.sgs.target 0.66 --runner.sgs.eval-steps 50`.
`sgs/*` logs report success, errors, entropy, valid/unvisited bins and outcome counts.

Model artifacts include `sgs.pt` in SGS mode. To restore only its sampling history,
pass `--runner.sgs-state-path /path/to/sgs.pt` with the same ordered clips and SGS
configuration. This is not full training resume: the target runner's existing
model/optimizer restore semantics are unchanged. State compatibility checks use
ordered clip names/lengths and sampler parameters, not full dataset hashing.
This runner is single-process; the source repository's DDP rank-sidecar protocol
is not introduced by this change.

## Public mjlab compatibility

The sibling dependency can be the public mjlab commit specified in the main
README plus `patches/mjlab-public.patch`, copied from docker150's local baseline.
The project-local motion adapter adds named multi-clip loading, clip-end handling,
external return sampling and optional SGS. Its differentiable FK reads the actual
robot scene, so P2 never uses the baseline patch's G1-only kinematics for training.
The same adapter provides G1's root velocities used by the velocity encoder.

## Checks

```bash
uv run pytest tests/test_fsq.py tests/test_sgs.py tests/test_p2.py -q
SUPERMIMIC_SIM_TEST=1 CUDA_VISIBLE_DEVICES= uv run pytest tests/test_p2_integration.py -q -s
uv run pytest -q
```

The opt-in smoke runs small CPU environments for both P2 representations, checks
the 50-step SGS reset, and performs world-model and policy updates. Synthetic
standing clips validate interfaces and gradients, not trained tracking quality.
