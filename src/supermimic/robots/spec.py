"""State schema shared by simulation, replay, networks and differentiable heads.

G1 is the unchanged default. Use SUPERMIMIC_ROBOT=t800 in a new process to
select T800. Checkpoints record the selection; mixed-robot loading is rejected.
"""
import os

ROBOT = os.environ.get("SUPERMIMIC_ROBOT", "g1").lower()
if ROBOT not in {"g1", "t800"}:
    raise ValueError(f"Unsupported SUPERMIMIC_ROBOT={ROBOT!r}")
NJ = {"g1": 29, "t800": 25}[ROBOT]
NQ = 7 + NJ
NV = 6 + NJ
STATE_DIM = NQ + NV
FEAT_DIM = STATE_DIM
DELTA_DIM = 12 + 2 * NJ
XI_DIM = NJ + 4
GOAL_FEAT_DIM = 2 * NJ + 10
OBS_FRAME_DIM = 3 * NJ + 6
ROOT_BODY = "pelvis" if ROBOT == "g1" else "LINK_BASE"
COM_BODY = "torso_link" if ROBOT == "g1" else "LINK_WAIST_YAW"


def load_robot_model():
    if ROBOT == "g1":
        from supermimic.data.motion_convert import load_g1_model
        return load_g1_model()
    from supermimic.robots.t800 import load_t800_model
    return load_t800_model()


def joint_names():
    if ROBOT == "g1":
        from supermimic.data.motion_convert import G1_JOINT_NAMES
        return G1_JOINT_NAMES
    from supermimic.robots.t800 import JOINT_NAMES
    return JOINT_NAMES


def validate_checkpoint_robot(config):
    saved = config.get("robot", "g1")
    if saved != ROBOT:
        raise ValueError(f"Checkpoint robot {saved!r} differs from process robot {ROBOT!r}")
