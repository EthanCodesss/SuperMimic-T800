"""Process-local, checkpoint-tagged action parameterization for a bounded ablation.

The audited source, actuator parameters, default pose, and reward coefficients
are unchanged. The raw-action Gaussian noise is unchanged too; its physical
amplitude changes with the scale, which is part of this intervention.
"""
import hashlib
import math
from pathlib import Path

MODES = ("rl_lab", "torque_kp")


def metadata(mode):
    from supermimic.robots.spec import ROBOT
    from supermimic.robots.t800 import ACTION_SCALE, ACTUATOR_GROUPS, JOINT_NAMES
    if ROBOT != "t800" or mode not in MODES:
        raise ValueError("Action-scale experiment requires T800 and a supported mode")
    values = dict(ACTION_SCALE)
    if mode == "torque_kp":
        for indices, kp, _kd, effort, _armature, _velocity in ACTUATOR_GROUPS:
            for index in indices:
                values[JOINT_NAMES[index]] = .25 * effort / kp
    assert tuple(values) == JOINT_NAMES
    assert all(math.isfinite(v) and v > 0 for v in values.values())
    return dict(mode=mode, formula="0.25 * effort_limit / Kp" if mode == "torque_kp" else "EngineAI RL Lab fixed scales",
                joint_names=list(JOINT_NAMES), scale=values,
                helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


def install(mode):
    from supermimic.envs import tracking_env_cfg
    spec = metadata(mode)
    tracking_env_cfg.ACTION_SCALE = dict(spec["scale"])
    return spec


def checkpoint_metadata(checkpoint):
    saved = checkpoint.get("execution", {}).get("action_parameterization")
    expected = metadata(saved["mode"] if saved else "rl_lab")
    if saved is not None and saved != expected:
        raise ValueError("Checkpoint action parameterization or helper provenance changed")
    return expected


def assert_runtime_scale(env, spec):
    import torch
    expected = torch.tensor(list(spec["scale"].values()), device=env.device)
    actual = env.action_manager.get_term("joint_pos")._scale
    torch.testing.assert_close(actual, expected.expand_as(actual), atol=0., rtol=1e-7)
