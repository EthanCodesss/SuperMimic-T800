"""Diagnostic reference-fed PD controller; this is NOT a learned policy.

Tests whether commanding the reference joint angles alone can maintain the
floating base. MuJoCo evolves the root freely after the initial reset.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--velocity-feedforward", action="store_true")
    args = parser.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = "t800"
    import numpy as np
    import torch
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.state_io import StateIO
    from supermimic.robots.t800 import ACTUATOR_GROUPS
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(args.motion)], num_envs=1, episode_length_s=1e5,
        action_noise_std=0., obs_noise=False, reset_noise=False, domain_randomization=False,
        sampling_mode="start", record_transitions=False,
    ), "cuda:0")
    try:
        obs, _ = env.reset()
        cmd = env.command_manager.get_term("motion")
        action = env.action_manager.get_term("joint_pos")
        io = StateIO(env)
        kd_over_kp = torch.zeros(25, device=env.device)
        for indices, kp, kd, *_ in ACTUATOR_GROUPS:
            kd_over_kp[list(indices)] = kd / kp
        trajectory, rewards, errors = [], [], []
        with torch.no_grad():
            for step in range(int(cmd.library.clip_len[0]) + 2):
                trajectory.append(io.read()[0, :32].cpu().numpy())
                target = cmd.joint_pos.clone()
                if args.velocity_feedforward:
                    target += kd_over_kp * cmd.joint_vel
                raw = (target - action.offset) / action.scale
                errors.append(float(torch.norm(cmd.robot_root_pos_w - cmd.root_pos_w)))
                obs, reward, term, timeout, _ = env.step(raw)
                rewards.append(float(reward[0]))
                if bool(term[0] | timeout[0]):
                    break
        result = {
            "controller": "reference-fed PD diagnostic, not learned",
            "velocity_feedforward": args.velocity_feedforward,
            "steps": len(rewards), "reference_frames": int(cmd.library.clip_len[0]),
            "completed_clip": bool(timeout[0]) and not bool(term[0]),
            "mean_root_error_m": float(np.mean(errors)), "normalized_return": float(np.mean(rewards) / .1),
            "termination_reasons": [n for n in env.termination_manager.active_terms if bool(env.termination_manager.get_term(n)[0])],
        }
        args.out.write_text(json.dumps(result, indent=2))
        np.savez_compressed(args.out.with_suffix(".trajectory.npz"), qpos=np.asarray(trajectory), fps=np.array([50.]))
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
