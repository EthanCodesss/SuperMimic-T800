"""Explicit anatomical reward mapping for the upstream single-motion comparison.

The locked source and validated physical robot are preserved. Install this
profile before importing the training loop; checkpoints carry its exact hash.
"""
import hashlib
from pathlib import Path

PROFILE='g1_semantic_v1'
G1_BODIES=(
    'pelvis','left_hip_roll_link','left_knee_link','left_ankle_roll_link',
    'right_hip_roll_link','right_knee_link','right_ankle_roll_link','torso_link',
    'left_shoulder_roll_link','left_elbow_link','left_wrist_yaw_link',
    'right_shoulder_roll_link','right_elbow_link','right_wrist_yaw_link',
)
T800_BODIES=(
    'LINK_BASE','LINK_HIP_ROLL_L','LINK_KNEE_PITCH_L','LINK_ANKLE_ROLL_L',
    'LINK_HIP_ROLL_R','LINK_KNEE_PITCH_R','LINK_ANKLE_ROLL_R','LINK_WAIST_YAW',
    'LINK_SHOULDER_ROLL_L','LINK_ELBOW_PITCH_L','LINK_WRIST_END_L',
    'LINK_SHOULDER_ROLL_R','LINK_ELBOW_PITCH_R','LINK_WRIST_END_R',
)


def metadata():
    from supermimic.robots.spec import ROBOT
    if ROBOT!='t800': raise ValueError('T800 task adapter requires a T800 process')
    return dict(profile=PROFILE,tracked_bodies=list(T800_BODIES),anchor_body='LINK_WAIST_YAW',
        root_body='LINK_BASE',g1_to_t800=dict(zip(G1_BODIES,T800_BODIES,strict=True)),
        head_control='All 25 joints retained; head is excluded only from the tracked-body average.',
        elbow_choice='Track the body at the elbow flexion hinge, as G1 elbow_link; T800 ELBOW_YAW is 13.2 cm farther down the forearm.',
        anchor_obs=False,helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def install():
    import sys
    if 'supermimic.training.loop' in sys.modules:
        raise RuntimeError('Install reward mapping before importing the training loop')
    from supermimic.envs import tracking_env_cfg
    spec=metadata()
    tracking_env_cfg.TRACKED_BODIES=T800_BODIES
    tracking_env_cfg.ANCHOR_BODY=spec['anchor_body']
    tracking_env_cfg.ROOT_BODY=spec['root_body']
    return spec


def checkpoint_metadata(checkpoint):
    expected=metadata()
    if checkpoint.get('execution',{}).get('task_adapter')!=expected:
        raise ValueError('Checkpoint reward mapping differs or is absent; use its original evaluator')
    return expected


def assert_runtime(env,spec,heads=None):
    cmd=env.command_manager.get_term('motion')
    assert tuple(cmd.cfg.body_names)==tuple(spec['tracked_bodies'])
    assert cmd.cfg.anchor_body_name==spec['anchor_body']
    assert cmd.cfg.root_body_name==spec['root_body']
    assert not env.cfg.observations['goal'].terms.get('anchor_pos')
    actual=[env.scene['robot'].body_names[i] for i in cmd.body_indexes.tolist()]
    assert actual==spec['tracked_bodies'],(actual,spec['tracked_bodies'])
    assert cmd.library.body_names==tuple(spec['tracked_bodies'])
    if heads is not None:
        assert heads.anchor_index==spec['tracked_bodies'].index(spec['anchor_body'])
        assert heads.root_fk_index==heads.fk.body_index(spec['root_body'])
        assert heads.tracked_idx.tolist()==heads.fk.body_indices(spec['tracked_bodies']).tolist()
