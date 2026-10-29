"""Historical diagnostic presets; NOT the strict reproduction entry point.

These presets reduce network/batch sizes and are not paper benchmark settings.
Use scripts/train_strict.py for source-aligned runs. Historical label-fix runs
also used the optional patch archived under patches/; it is no longer applied.
"""
import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("g1", "t800"), required=True)
    parser.add_argument("--motion-dir", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--iterations", type=int, default=160)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--training", action="store_true", help="Use larger batches/horizon for learning")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--horizon", type=int, help="Override imagination horizon")
    parser.add_argument("--anchor-obs", action=argparse.BooleanOptionalAction, default=None,
                        help="Override robot default; use for matched G1/T800 controls")
    parser.add_argument("--no-critic", action="store_true", help="Diagnostic ablation of terminal value bootstrap")
    args = parser.parse_args()
    print("[DIAGNOSTIC] Reduced recipe, not upstream reproduction. Strict entry: scripts/train_strict.py", flush=True)
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    import torch
    torch.set_num_threads(4)
    from supermimic.config import TrainConfig
    from supermimic.training.loop import Trainer

    cfg = TrainConfig(run_name=args.run_name, resume=args.resume)
    cfg.data.npz_dir = args.motion_dir
    cfg.env.num_envs = args.num_envs
    cfg.env.replay_capacity = 256 if args.training else 96
    cfg.env.episode_length_s = 10.
    cfg.env.domain_randomization = False
    cfg.env.reset_noise = False
    cfg.env.obs_noise = False
    cfg.env.action_noise_std = .1
    cfg.env.sampling_mode = "uniform"
    cfg.env.anchor_obs = args.robot == "t800"
    if args.anchor_obs is not None:
        cfg.env.anchor_obs = args.anchor_obs
    cfg.world_model.hidden = (512, 512, 512) if args.training else (256, 256, 256)
    cfg.policy.encoder_hidden = (512, 256, 128) if args.training else (256, 128)
    cfg.policy.decoder_hidden = cfg.policy.encoder_hidden
    cfg.wm_trainer.batch_size = 256 if args.training else 64
    cfg.wm_trainer.batches_per_update = 4 if args.training else 1
    cfg.policy_trainer.batch_size = 256 if args.training else 64
    cfg.policy_trainer.horizon = 16 if args.training else 8
    if args.horizon is not None:
        cfg.policy_trainer.horizon = args.horizon
    cfg.policy_trainer.critic_epochs = 2
    cfg.policy_trainer.critic_minibatches = 2
    cfg.policy_trainer.obs_noise = False
    cfg.policy_trainer.use_critic = not args.no_critic
    cfg.wm_trainer.compile = "default" if args.compile else None
    cfg.policy_trainer.compile = "default" if args.compile else None
    cfg.schedule.total_iterations = args.iterations
    cfg.schedule.checkpoint_interval = 1000 if args.training else 80
    cfg.schedule.log_interval = 50 if args.training else 20
    cfg.schedule.warmup_iterations = 40
    cfg.schedule.lr_decay_fraction = 1.
    # Keep the LR useful for short tests and incremental training extensions.
    cfg.schedule.lr_final_ratio = .5
    output = Path("configs") / f"{args.run_name}.json"
    output.write_text(json.dumps(asdict(cfg), indent=2, default=str))
    trainer = Trainer(cfg)
    try:
        if args.resume is None:
            trainer.save("ckpt_initial.pt")
        trainer.run()
        stats = {
            "robot": args.robot, "iteration": trainer.iteration,
            "updates": trainer.num_updates, "resumed_from": str(args.resume) if args.resume else None,
            "peak_cuda_memory_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "gpu_memory_measurement": "Torch allocator only; excludes MuJoCo Warp and CUDA context",
            "replay_rows": trainer.buffer.size,
            "world_model_online_nse_by_horizon": trainer.wm_trainer.online_error(trainer.buffer, n=128, k=8).tolist(),
        }
        (trainer.log_dir / "run_summary.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats, indent=2))
    finally:
        trainer.env.close()
        if trainer.writer:
            trainer.writer.close()


if __name__ == "__main__":
    main()
