"""Astro P2 V3 asset and motor-family control configuration."""

import json
import math
from pathlib import Path

import mujoco

_ASSETS = Path(__file__).resolve().parents[1] / "assets/p2"
_XML = _ASSETS / "mjcf/astro_p2_supermimic.xml"
_CONTRACT = json.loads((_ASSETS / "p2_deploy_contract.json").read_text())
P2_JOINT_NAMES = tuple(_CONTRACT["joint_names"][:-1])
P2_FOOT_GEOMS = r"^(left|right)_foot_capsule_collision[1-8]$"

_DEFAULT_ANGLES = {
  "left_hip_pitch_joint": -0.312,
  "right_hip_pitch_joint": -0.312,
  "left_knee_joint": 0.669,
  "right_knee_joint": 0.669,
  "left_ankle_pitch_joint": -0.363,
  "right_ankle_pitch_joint": -0.363,
  "left_shoulder_pitch_joint": 0.2,
  "right_shoulder_pitch_joint": 0.2,
  "left_shoulder_roll_joint": 0.2,
  "right_shoulder_roll_joint": -0.2,
  "left_elbow_joint": 0.6,
  "right_elbow_joint": 0.6,
}


def load_p2_model() -> mujoco.MjModel:
  return mujoco.MjModel.from_xml_path(str(_XML))


def _p2_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(_XML))
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="frame__imu_2_link",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="frame__imu_2_link",
  )
  return spec


# Attachment motor families: joint-side inertia, natural frequency in Hz,
# and measured motor torque-speed envelope (knee, no-load speed, peak torque).
_MOTOR_39 = (0.002387232, 12.0, (11.610585068918155, 26.897164278724524, 10.0))
_MOTOR_50 = (0.0087362778817, 10.0, (14.975531892552269, 34.51807695681387, 30.0))
_MOTOR_85 = (0.04416342683946, 10.0, (7.329846222734013, 26.754377942701183, 130.0))

# Each group shares a command delay. Parallel groups are ordered pitch, roll.
_GROUPS = (
  (
    (
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
    ),
    _MOTOR_85,
    (130,) * 8,
    23.040,
    False,
  ),
  (
    ("left_ankle_pitch_joint", "left_ankle_roll_joint"),
    _MOTOR_50,
    (60, 45),
    26.180,
    True,
  ),
  (
    ("right_ankle_pitch_joint", "right_ankle_roll_joint"),
    _MOTOR_50,
    (60, 45),
    26.180,
    True,
  ),
  (("waist_yaw_joint",), _MOTOR_85, (130,), 13.040, False),
  (("waist_pitch_joint", "waist_roll_joint"), _MOTOR_50, (60, 50), 26.180, True),
  (
    tuple(
      f"{side}_{joint}_joint"
      for side in ("left", "right")
      for joint in (
        "shoulder_pitch",
        "shoulder_roll",
        "shoulder_yaw",
        "elbow",
        "wrist_roll",
      )
    ),
    _MOTOR_50,
    (30,) * 10,
    26.180,
    False,
  ),
  (
    tuple(
      f"{side}_wrist_{axis}_joint"
      for side in ("left", "right")
      for axis in ("pitch", "yaw")
    ),
    _MOTOR_39,
    (10,) * 4,
    20.940,
    False,
  ),
)


def make_p2_robot():
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

  actuators = []
  scales = {}
  for names, (inertia, hz, _curve), efforts, _speed, parallel in _GROUPS:
    omega = 2 * math.pi * hz
    joint_inertia = inertia * (2 if parallel else 1)
    kp = joint_inertia * omega**2
    kd = 4 * joint_inertia * omega  # damping ratio 2
    scales.update(
      {name: 0.25 * effort / kp for name, effort in zip(names, efforts, strict=True)}
    )
    actuators.extend(
      BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=kp,
        damping=kd,
        effort_limit=effort,
      )
      for name, effort in zip(names, efforts, strict=True)
    )

  cfg = EntityCfg(
    spec_fn=_p2_spec,
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, 0.645),
      joint_pos={name: _DEFAULT_ANGLES.get(name, 0.0) for name in P2_JOINT_NAMES},
    ),
    articulation=EntityArticulationInfoCfg(
      actuators=tuple(actuators), soft_joint_pos_limit_factor=0.9
    ),
    sort_actuators=True,
  )
  return cfg, scales
