"""Install and validate the complete, versioned T800 training/evaluation profile."""
import hashlib
from pathlib import Path


def install():
    from scripts.t800_action_scaling import install as install_scale
    from scripts.t800_task_adapter import install as install_task
    from scripts.fk_compile_compat import install as install_fk
    return dict(action_parameterization=install_scale('torque_kp'),
                task_adapter=install_task(), fk_compile_compat=install_fk(),
                t800_profile_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def validate_checkpoint(checkpoint, profile):
    for key, value in profile.items():
        if checkpoint.get('execution', {}).get(key) != value:
            raise ValueError(f'Checkpoint T800 profile is absent or differs: {key}')


def assert_runtime(env, profile, heads=None):
    from scripts.t800_action_scaling import assert_runtime_scale
    from scripts.t800_task_adapter import assert_runtime as assert_task
    assert_runtime_scale(env, profile['action_parameterization'])
    assert_task(env, profile['task_adapter'], heads)
