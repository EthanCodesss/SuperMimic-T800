"""Checkpoint-aware action-scale variant of evaluate_tracking.py.

Evaluate one clip from its first frame. No updates are made to the checkpoint;
world-model errors use new evaluation transitions, separate from training replay.
"""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("g1", "t800"), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-noise", action="store_true",
                        help="Use the original reset perturbations for a robustness episode")
    parser.add_argument("--wm-test-windows", type=Path, help="Optional fixed transition windows for checkpoint comparisons")
    parser.add_argument("--wm-horizon", type=int, default=8,
                        help="Length of held-out model rollouts; use 32 to match policy imagination")
    args = parser.parse_args()
    if args.wm_horizon < 1:
        parser.error("wm-horizon must be positive")
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    import numpy as np
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    from supermimic.envs import SuperMimicEnvCfgArgs, make_env
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.spec import NQ, load_robot_model
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.world_model import WorldModelCfg, WorldModel
    from supermimic.kinematics.rotations import heading_yaw, wrap_to_pi
    from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
    with np.load(args.motion) as motion_data:
        reference_frames = len(motion_data["joint_pos"])
    ck = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    from scripts.t800_action_scaling import checkpoint_metadata, install, assert_runtime_scale
    action_spec = checkpoint_metadata(ck)
    install(action_spec["mode"])
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, "cuda")
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(args.motion)], num_envs=1, episode_length_s=1e5,
        future_frames=cfg["env"]["future_frames"], anchor_obs=cfg["env"].get("anchor_obs", False),
        action_noise_std=0., obs_noise=False, reset_noise=args.reset_noise, domain_randomization=False,
        sampling_mode="start", record_transitions=True,
        replay_capacity=max(1024, reference_frames + args.wm_horizon + 2), seed=args.seed,
    ), device="cuda:0")
    assert_runtime_scale(env, action_spec)
    from mjlab.utils.torch import configure_torch_backends
    precision = ck.get("execution", {}).get("backend_precision", "tf32")
    configure_torch_backends(allow_tf32=precision == "tf32")
    try:
        rec = env.recorder_manager.get_term("replay")
        cmd = env.command_manager.get_term("motion")
        robot = env.scene["robot"]
        fk = G1Kinematics(load_robot_model(), device="cuda")
        heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY,
                              robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        wm_dict = dict(cfg["world_model"])
        wm_dict["state_spec"] = StateSpecCfg(**wm_dict["state_spec"])
        wm = WorldModel(WorldModelCfg(**wm_dict), heads, "cuda")
        wm.load_state_dict(ck["wm"])
        wm.eval()
        # Capture the exact reward-time state before an automatic reset. This
        # diagnostic does not feed back into observations, actions, or training.
        reward_terms, velocity_snapshots = [], []
        compute_reward = env.reward_manager.compute
        def capture_reward(dt):
            value = compute_reward(dt)
            reward_terms.append(env.reward_manager.term_rewards[0].clone())
            velocity_snapshots.append(tuple(x[0].clone() for x in (
                cmd.anchor_quat_w, cmd.robot_anchor_quat_w,
                cmd.body_lin_vel_w, cmd.robot_body_lin_vel_w,
                cmd.body_ang_vel_w, cmd.robot_body_ang_vel_w)))
            return value
        env.reward_manager.compute = capture_reward
        obs, _ = env.reset()
        qpos, root_error, joint_sq_error, body_error, reward = [], [], [], [], []
        reference_qpos, target_frames, anchor_snapshots = [], [], []
        raw_actions = []
        terminated = False
        termination_reasons = []
        with torch.no_grad():
            for step in range(int(cmd.library.clip_len[0]) + 2):
                qpos.append(rec.state_io.read()[0, :NQ].cpu().numpy())
                ref = cmd.library.ref(cmd.frame)
                reference_qpos.append(torch.cat([ref.root_pos_w, ref.root_quat_w, ref.joint_pos], -1)[0].cpu().numpy())
                target_frames.append(int(cmd.frame[0]))
                root_error.append(float(torch.norm(cmd.robot_root_pos_w - cmd.root_pos_w)))
                anchor_snapshots.append(tuple(x[0].clone() for x in (
                    cmd.robot_anchor_pos_w, cmd.anchor_pos_w, cmd.robot_anchor_quat_w, cmd.anchor_quat_w)))
                joint_sq_error.append(float((cmd.robot_joint_pos - cmd.joint_pos).square().mean()))
                body_error.append(float(torch.norm(cmd.robot_body_pos_w - cmd.body_pos_relative_w, dim=-1).mean()))
                act = policy(obs["policy"], obs["goal"])
                raw_actions.append(act[0].cpu().numpy())
                obs, r, term, timeout, _ = env.step(act)
                reward.append(float(r[0]))
                if bool(term[0] | timeout[0]):
                    terminated = bool(term[0])
                    termination_reasons = [name for name in env.termination_manager.active_terms if bool(env.termination_manager.get_term(name)[0])]
                    break
            else:
                raise RuntimeError("Expected clip end or termination")
            errors = None
            fixed_errors = None
            if rec.buffer.window_mask(args.wm_horizon).any():
                windows = rec.buffer.sample_windows(128, args.wm_horizon)
                errors = wm.rollout_error(windows.states, windows.a_exec, windows.xi).tolist()
                args.out.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "states": windows.states.cpu(), "actions": windows.a_exec.cpu(),
                    "xi": windows.xi.cpu(), "robot": args.robot,
                    "source_checkpoint": str(args.checkpoint), "source_motion": str(args.motion),
                    "action_parameterization": action_spec,
                }, args.out.with_suffix(".wm_windows.pt"))
            if args.wm_test_windows is not None:
                fixed = torch.load(args.wm_test_windows, map_location="cuda", weights_only=True)
                if fixed["robot"] != args.robot:
                    raise ValueError("Fixed test windows belong to a different robot")
                fixed_mode = fixed.get("action_parameterization", {}).get("mode", "rl_lab")
                if fixed_mode != action_spec["mode"]:
                    raise ValueError("Fixed WM actions must be re-encoded for the checkpoint's scale")
                fixed_errors = wm.rollout_error(fixed["states"], fixed["actions"], fixed["xi"]).tolist()
        d = rec.diagnostics.cpu().numpy()
        # Keep extra diagnostic geometry outside the active control loop.
        with torch.no_grad():
            ap, ar, aq, aqr = (torch.stack([x[i] for x in anchor_snapshots]).cpu() for i in range(4))
            xy_error = (ap[:, :2] - ar[:, :2]).norm(dim=-1).numpy()
            yaw_error = wrap_to_pi(heading_yaw(aq) - heading_yaw(aqr)).numpy()
            rq, bq, rv, bv, rw, bw = (torch.stack([x[i] for x in velocity_snapshots]).cpu() for i in range(6))
            rotation = yaw_quat(quat_mul(rq, quat_inv(bq)))[:, None, :].expand(*bv.shape[:-1], 4)
            ev = (rv - quat_apply(rotation, bv)).square().sum(-1).mean(-1)
            ew = (rw - quat_apply(rotation, bw)).square().sum(-1).mean(-1)
            aligned_velocity_rewards = env.step_dt * torch.stack([torch.exp(-ev), torch.exp(-ew / 3.14**2)], -1)
        term_array = torch.stack(reward_terms).cpu().numpy()
        term_names = list(env.reward_manager.active_terms)
        term_mean = dict(zip(term_names, term_array.mean(0).astype(float).tolist()))
        if not np.allclose(term_array.sum(-1), reward, atol=2e-7, rtol=2e-6):
            raise AssertionError("Per-term rewards do not sum to the observed step reward")
        yaw_deg = np.rad2deg(yaw_error)
        time_s = np.arange(len(reward)) * env.step_dt
        def first_crossing(values, threshold):
            hit = np.flatnonzero(np.asarray(values) > threshold)
            return float(time_s[hit[0]]) if len(hit) else None
        result = {
            "action_parameterization": action_spec,
            "robot": args.robot, "checkpoint": str(args.checkpoint), "motion": str(args.motion),
            "checkpoint_iteration": ck.get("iteration"),
            "checkpoint_updates": ck.get("num_updates"), "steps": len(reward),
            "reference_frames": int(cmd.library.clip_len[0]),
            "completed_clip": not terminated and "clip_end" in termination_reasons,
            "completion_fraction": min(1., len(reward) / max(1, int(cmd.library.clip_len[0]) - 1)),
            "root_height_or_tilt_failure": terminated, "termination_reasons": termination_reasons,
            "mean_root_position_error_m": float(np.mean(root_error)),
            "final_root_position_error_m": float(root_error[-1]),
            "mean_anchor_xy_error_m": float(np.mean(xy_error)),
            "final_anchor_xy_error_m": float(xy_error[-1]),
            "mean_abs_yaw_error_deg": float(np.mean(np.abs(yaw_deg))),
            "final_yaw_error_deg": float(yaw_deg[-1]),
            "first_xy_error_over_0p3m_s": first_crossing(xy_error, .3),
            "first_abs_yaw_error_over_15deg_s": first_crossing(np.abs(yaw_deg), 15.),
            "joint_position_rmse_rad": float(np.sqrt(np.mean(joint_sq_error))),
            "mean_anchor_aligned_body_position_error_m": float(np.mean(body_error)),
            "normalized_return": float(np.mean(reward) / .1),
            "normalized_full_clip_return": float(np.sum(reward) / (.1 * reference_frames)),
            "full_clip_return_definition": "Sum of observed rewards divided by 0.1 times full reference frame count; unreached steps contribute zero",
            "mean_step_reward_terms": term_mean,
            "reward_fraction_of_term_max": {k: v / (.01 if 'global_root' in k else .02)
                                            for k, v in term_mean.items() if k.startswith('motion_')},
            "diagnostic_yaw_aligned_velocity_reward": dict(zip(
                ['linear', 'angular'], aligned_velocity_rewards.mean(0).tolist())),
            "hard_joint_limit_fraction": float(d[2] / max(d[1], 1)),
            "soft_joint_limit_fraction": float(d[3] / max(d[1], 1)),
            "torque_saturation_fraction": float(d[4] / max(d[5], 1)),
            "diagnostics_sampling": "last physics substep of each 50 Hz control step; includes terminal step",
            "world_model_eval_nse_by_horizon": errors,
            "world_model_fixed_test_nse_by_horizon": fixed_errors,
            "world_model_fixed_test_source": str(args.wm_test_windows) if args.wm_test_windows else None,
            "world_model_error_data": "new deterministic rollout, no parameter updates, normalized squared state error",
            "evaluation_episodes": 1,
            "evaluation_seed": args.seed, "reset_noise": args.reset_noise,
            "matmul_precision": torch.backends.cuda.matmul.fp32_precision,
            "anchor_obs": cfg["env"].get("anchor_obs", False),
            "trajectory_alignment": "Pre-action robot state and exact command reference frame saved together; rewards are from the subsequent step",
            "raw_action_rms_per_joint": np.sqrt(np.mean(np.square(raw_actions), axis=0)).astype(float).tolist(),
            "physical_target_offset_rms_per_joint_rad": (np.sqrt(np.mean(np.square(raw_actions), axis=0)) * np.asarray(list(action_spec["scale"].values()))).tolist(),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2))
        np.savez_compressed(args.out.with_suffix(".trajectory.npz"), qpos=np.asarray(qpos), fps=np.array([50.]),
                            reference_qpos=np.asarray(reference_qpos), target_frames=np.asarray(target_frames),
                            time_s=time_s, root_error_m=np.asarray(root_error), joint_mse=np.asarray(joint_sq_error),
                            xy_error_m=np.asarray(xy_error), yaw_error_deg=yaw_deg,
                            body_error_m=np.asarray(body_error), reward_terms=term_array,
                            reward_term_names=np.asarray(term_names))
        # Action diagnostics are a separate small artifact; preserve the trajectory schema.
        np.savez_compressed(args.out.with_suffix(".actions.npz"), raw_actions=np.asarray(raw_actions),
                            scales=np.asarray(list(action_spec["scale"].values())),
                            joint_names=np.asarray(action_spec["joint_names"]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        qr, qf = np.asarray(qpos), np.asarray(reference_qpos)
        axes[0, 0].plot(qf[:, 0], qf[:, 1], label="Reference")
        axes[0, 0].plot(qr[:, 0], qr[:, 1], label="Robot")
        axes[0, 0].set(xlabel="x (m)", ylabel="y (m)", title="World trajectory", aspect="equal")
        axes[0, 1].plot(time_s, root_error, label="Root position (m)")
        axes[0, 1].plot(time_s, body_error, label="Aligned bodies (m)")
        axes[0, 1].set(xlabel="Time (s)", title="Position error")
        axes[1, 0].plot(time_s, yaw_deg, label="Yaw error (deg)")
        axes[1, 0].set(xlabel="Time (s)", title="Heading error")
        for i, name in enumerate(term_names):
            axes[1, 1].plot(time_s + env.step_dt, term_array[:, i], label=name.replace("motion_", ""))
        axes[1, 1].set(xlabel="Time (s)", title="Weighted step rewards")
        for ax in axes.flat:
            ax.legend(fontsize=7)
            ax.grid(alpha=.2)
        fig.suptitle(f"{args.checkpoint.parent.name}: updates={ck.get('num_updates')} seed={args.seed} reset_noise={args.reset_noise}")
        fig.savefig(args.out.with_suffix(".png"), dpi=130)
        plt.close(fig)
        print(json.dumps(result, indent=2))
    finally:
        env.close()


if __name__ == "__main__":
    main()
