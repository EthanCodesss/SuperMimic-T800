"""Versioned T800 anchor observations + fixed-head control experiment."""
import hashlib
from pathlib import Path


def install():
    from scripts.t800_fk_profile import install as base_install
    from scripts.t800_fixed_head import install as head_install
    profile = base_install()
    profile['task_adapter'] = dict(profile['task_adapter'], anchor_obs=True,
        head_control='23 active body outputs; head PD targets fixed at default; 25-D state/action schema retained.')
    profile['head_control'] = head_install()
    profile['t800_profile_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return profile


def validate_checkpoint(checkpoint, profile):
    from scripts.t800_fk_profile import validate_checkpoint as base_validate
    base_validate(checkpoint, profile)
    cfg = checkpoint['config']
    assert cfg['env']['anchor_obs'] and cfg['policy_trainer']['anchor_obs']
    assert cfg['policy']['goal_dim'] == 609


def assert_runtime(env, profile, heads=None):
    from scripts.t800_action_scaling import assert_runtime_scale
    from scripts.t800_fixed_head import assert_runtime as assert_head
    assert_runtime_scale(env, profile['action_parameterization'])
    assert_head(env, profile['head_control'])
    spec = profile['task_adapter']
    cmd = env.command_manager.get_term('motion')
    assert tuple(cmd.cfg.body_names) == tuple(spec['tracked_bodies'])
    assert cmd.cfg.anchor_body_name == spec['anchor_body']
    assert cmd.cfg.root_body_name == spec['root_body']
    assert list(env.cfg.observations['goal'].terms) == ['future_frames', 'anchor_pos', 'anchor_ori']
    assert [env.scene['robot'].body_names[i] for i in cmd.body_indexes.tolist()] == spec['tracked_bodies']
    assert cmd.library.body_names == tuple(spec['tracked_bodies'])
    if heads is not None:
        assert heads.anchor_index == spec['tracked_bodies'].index(spec['anchor_body'])
        assert heads.root_fk_index == heads.fk.body_index(spec['root_body'])
        assert heads.tracked_idx.tolist() == heads.fk.body_indices(spec['tracked_bodies']).tolist()
