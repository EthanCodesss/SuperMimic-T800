"""EngineAI serial T800, pinned GMR geometry + pinned RL Lab control parameters.

MuJoCo implicit PD, 200 Hz simulation / 50 Hz control. The deterministic
initial baseline omits the Isaac Lab task's randomized 1–3 step delay.
Nominal velocity ratings are recorded, not imposed as hard velocity clamps.
"""
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

REPO_ROOT = Path(__file__).resolve().parents[3]
XML = REPO_ROOT / "vendor/engineai_gmr/assets/engineai_t800/xml/serial_t800.xml"
JOINT_NAMES = (
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L", "J06_HIP_PITCH_R", "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_TORSO_YAW", "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R", "J23_HEAD_PITCH", "J24_HEAD_YAW",
)
TRACKED_BODIES = (
    "LINK_BASE", "LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R", "LINK_KNEE_PITCH_R", "LINK_ANKLE_ROLL_R", "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_YAW_L", "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_YAW_R", "LINK_WRIST_END_R", "LINK_HEAD_YAW",
)
ANCHOR_BODY = ROOT_BODY = "LINK_BASE"
FOOT_GEOM_REGEX = r"^LINK_FOOT_[LR]_collision_[0-9]+$"
DEFAULT_POS = dict.fromkeys(JOINT_NAMES, 0.0)
for side, hip, knee, ankle, shoulder, elbow in (
    ("L", 0, 3, 4, 14, 16), ("R", 6, 9, 10, 19, 21),
):
    DEFAULT_POS[JOINT_NAMES[hip]] = -0.06
    DEFAULT_POS[JOINT_NAMES[knee]] = 0.12
    DEFAULT_POS[JOINT_NAMES[ankle]] = -0.06
    DEFAULT_POS[JOINT_NAMES[shoulder]] = 0.15 if side == "L" else -0.15
    DEFAULT_POS[JOINT_NAMES[elbow]] = -0.25
ACTION_SCALE = dict(zip(JOINT_NAMES, (
    .5, .2, .2, .5, .5, .2, .5, .2, .2, .5, .5, .2, .2,
    .2, .2, .05, .2, .05, .2, .2, .05, .2, .05, .2, .2,
), strict=True))
# indices, kp, kd, torque limit Nm, rotor armature, nominal velocity rad/s
ACTUATOR_GROUPS = (
    ((0, 3, 6, 9), 180., 5., 415. * .95, .2427264, 25.96),
    ((1, 7), 100., 3., 370., .14110848, 25.31),
    ((2, 8, 12), 100., 3., 222., .0448737, 23.19),
    ((4, 5, 10, 11), 40., 2., 160., .0354625, 33.51),
    ((13, 14, 15, 16, 18, 19, 20, 21), 50., .3, 160., .0354625, 33.51),
    ((17, 22, 23, 24), 50., .3, 52., .00671625, 35.2),
)


def get_spec():
    spec = mujoco.MjSpec.from_file(str(XML))
    # Replace native torque motors with mjlab's implicit PD actuators.
    for actuator in list(spec.actuators):
        spec.delete(actuator)
    for sensor in list(spec.sensors):
        spec.delete(sensor)
    for light in list(spec.lights):
        spec.delete(light)
    for body in spec.bodies:
        for index, geom in enumerate(body.geoms):
            if geom.group == 3:
                geom.name = f"{body.name}_collision_{index}"
    return spec


def load_t800_model():
    model = get_spec().compile()
    names = tuple(model.joint(i).name for i in range(1, model.njnt))
    if names != JOINT_NAMES or (model.nq, model.nv) != (32, 31):
        raise ValueError(f"T800 model schema changed: {names}")
    return model


def get_t800_robot_cfg():
    actuators = tuple(BuiltinPositionActuatorCfg(
        target_names_expr=tuple(JOINT_NAMES[i] for i in indices),
        stiffness=kp, damping=kd, effort_limit=limit, armature=armature,
        frictionloss=0.0, viscous_damping=0.0,
    ) for indices, kp, kd, limit, armature, velocity in ACTUATOR_GROUPS)
    return EntityCfg(
        spec_fn=get_spec,
        init_state=EntityCfg.InitialStateCfg(pos=(0., 0., 1.06), joint_pos=DEFAULT_POS.copy()),
        articulation=EntityArticulationInfoCfg(actuators=actuators, soft_joint_pos_limit_factor=.9),
        collisions=(CollisionCfg(
            geom_names_expr=(".*_collision_[0-9]+",), contype=0, conaffinity=1,
            condim=3, priority=1, friction=(1.0, .005, .0001),
        ),),
    )
