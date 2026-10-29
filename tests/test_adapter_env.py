"""Physical adapter checks: joint actions, analytic observations/rewards, replay."""
from pathlib import Path

import numpy as np
import pytest
import torch

from supermimic.robots.spec import NJ, NQ, STATE_DIM, XI_DIM, ROBOT, load_robot_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_physical_adapter_and_analytic_heads():
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, OBS_TERMS, analytic_observation
    from supermimic.world_model.wm_trainer import reference_at
    from mjlab.tasks.tracking import mdp as tracking
    path = Path("data/npz/t800_lafan1/walk2_short.npz" if ROBOT == "t800" else "data/npz/g1_smoke/walk1_short.npz")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(path)], num_envs=4, replay_capacity=64,
        obs_noise=False, reset_noise=False, domain_randomization=False,
        action_noise_std=0., sampling_mode="start", anchor_obs=True,
    ), "cuda:0")
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        robot = env.scene["robot"]
        rec = env.recorder_manager.get_term("replay")
        cmd = env.command_manager.get_term("motion")
        fk = G1Kinematics(load_robot_model(), device=env.device)
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        obs, _ = env.reset()
        for step in range(24):
            previous_policy_obs = obs["policy"][0:1].clone()
            previous_goal = obs["goal"][0:1].clone()
            written_row = rec.buffer.ptr[0].clone()
            # Different desired positions on each DOF catch actuator-order errors.
            action = torch.linspace(-.1, .1, NJ, device=env.device).repeat(4, 1)
            obs, _, _, _, _ = env.step(action)
            state = rec.state_io.read()
            assert state.shape == (4, STATE_DIM)
            assert rec.ctx.read().shape == (4, XI_DIM)
            if env.reset_buf.any():
                continue
            ours = analytic_observation(state, action, rec.ctx.read(), robot.data.default_joint_pos[0])
            start = 0
            for name, size in OBS_TERMS:
                actual = obs["policy"][:, start * 10:(start + size) * 10].reshape(4, 10, size)[:, -1]
                torch.testing.assert_close(ours[:, start:start+size], actual, atol=3e-5, rtol=1e-5)
                start += size
            bodies = heads.robot_bodies(state)
            torch.testing.assert_close(bodies["pos"], cmd.robot_body_pos_w - env.scene.env_origins[:, None], atol=3e-5, rtol=1e-5)
            rewards = heads.reward(state, action, env.action_manager.prev_action, reference_at(cmd.library, cmd.frame))
            expected = tracking.motion_relative_body_position_error_exp(env, "motion", .3)
            torch.testing.assert_close(rewards["motion_body_pos"] / .02, expected, atol=3e-5, rtol=1e-5)
            if step >= 10:
                from supermimic.world_model.heads import flatten_history_term_major, anchor_relative_obs
                sample = rec.buffer._sample_from_mask
                rec.buffer._sample_from_mask = lambda mask, n: (
                    torch.tensor([0], device=env.device), written_row.reshape(1))
                try:
                    start = rec.buffer.sample_starts(1, 10)
                finally:
                    rec.buffer._sample_from_mask = sample
                reconstructed = flatten_history_term_major(analytic_observation(
                    start.hist_states, start.hist_actions, start.xi,
                    robot.data.default_joint_pos[0]))
                torch.testing.assert_close(reconstructed, previous_policy_obs, atol=3e-5, rtol=1e-5)
                body = heads.robot_bodies(start.state)
                ref = reference_at(cmd.library, start.frame)
                ai = heads.anchor_index
                anchor = anchor_relative_obs(body["pos"][:, ai], body["quat"][:, ai],
                                            ref.body_pos_w[:, ai], ref.body_quat_w[:, ai])
                goal = torch.cat([cmd.library.goal_window(start.frame, 10), anchor], dim=-1)
                torch.testing.assert_close(goal, previous_goal, atol=3e-5, rtol=1e-5)
        windows = rec.buffer.sample_windows(16, 8)
        assert windows.states.shape == (9, 16, STATE_DIM)
        assert windows.a_exec.shape == (8, 16, NJ)
        assert torch.isfinite(windows.states).all()
        assert rec.diagnostics[0] == 4 * 24
    finally:
        env.close()
