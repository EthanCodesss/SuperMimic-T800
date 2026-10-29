"""Opt-in fixed head targets, consistently applied to actor, simulation and replay.

Keep the 25-D schema and physical head dynamics. The last two actor outputs have
zero gradient; manager history and executed replay actions contain zeros there.
Draw the original execution noise before masking so body noise draws are intact.
"""
import hashlib
from pathlib import Path


def mask_action(action):
    import torch
    if action.shape[-1] != 25:
        raise ValueError('Fixed T800 head expects the original 25-D action schema')
    return torch.cat((action[..., :23], torch.zeros_like(action[..., 23:])), dim=-1)


def install():
    from supermimic.robots.spec import ROBOT
    from supermimic.robots.t800 import JOINT_NAMES
    from supermimic.policy.nets import Policy
    from supermimic.envs.actions import NoisyJointPositionAction
    from mjlab.managers.action_manager import ActionManager
    assert ROBOT == 't800'
    assert JOINT_NAMES[23:] == ('J23_HEAD_PITCH', 'J24_HEAD_YAW')
    metadata = dict(profile='t800_fixed_head_default_v1', joint_names=list(JOINT_NAMES[23:]),
        joint_indices=[23, 24], target='robot default_joint_pos, via unchanged PD',
        policy_action_dim=25, effective_controlled_action_dim=23,
        execution_noise='Original Gaussian draw; head entries discarded before target and replay',
        state='All 25 physical joint positions and velocities retained; no state clamping',
        reward='Head excluded from tracked bodies; original penalties retained on masked actions and actual state',
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if hasattr(Policy, '_t800_fixed_head_metadata'):
        assert Policy._t800_fixed_head_metadata == metadata
        return metadata
    policy_forward = Policy.forward
    manager_process = ActionManager.process_action
    joint_process = NoisyJointPositionAction.process_actions

    def fixed_policy(self, obs_hist, goal, interface='motion'):
        return mask_action(policy_forward(self, obs_hist, goal, interface))

    def fixed_manager(self, action):
        # Mask before the manager stores action/prev_action or the recorder sees it.
        return manager_process(self, mask_action(action))

    def fixed_joint(self, actions):
        joint_process(self, mask_action(actions))
        self._executed_raw = mask_action(self._executed_raw)
        # Preserve the ordinary affine mapping/clip for the body; head target is
        # exactly the default offset even when execution noise is enabled.
        self._processed_actions[..., 23:] = self._offset[..., 23:]

    Policy.forward = fixed_policy
    ActionManager.process_action = fixed_manager
    NoisyJointPositionAction.process_actions = fixed_joint
    Policy._t800_fixed_head_metadata = metadata
    return metadata


def assert_runtime(env, metadata):
    import torch
    robot = env.scene['robot']
    assert list(robot.joint_names[23:]) == metadata['joint_names']
    term = env.action_manager.get_term('joint_pos')
    torch.testing.assert_close(term._offset[..., 23:], robot.data.default_joint_pos[..., 23:], rtol=0, atol=0)
    assert term.cfg.use_default_offset
