"""Isolated copy of train_memory.py for the T800 action-scale ablation.

Uses the audited T800 source with an explicit process-local action-scale override
and the existing activation storage optimization. PolicyTrainer.update is inherited unchanged. Stopping after a number
of updates does not shorten the LR schedule.
Resume follows upstream semantics: replay is refilled, RNG is not restored.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_strict import select_source, sha, verify_sources, write_json


def tensor_digest(tree):
    """Content digest for exact model/optimizer reload checks, independent of zip encoding."""
    import torch
    digest = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            value = value.detach().contiguous().cpu()
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                visit(value[key])
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
        else:
            digest.update(repr(value).encode())
    visit(tree)
    return digest.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("g1", "t800"), required=True)
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--clips", nargs="+")
    p.add_argument("--run-name", required=True)
    stop = p.add_mutually_exclusive_group()
    stop.add_argument("--stop-after-updates", type=int, default=3,
                      help="Additional updates this run; 0 runs to the original 320k step limit")
    stop.add_argument("--stop-at-iteration", type=int,
                      help="Stop at this absolute control-step milestone without changing the LR schedule")
    p.add_argument("--extra-checkpoint-interval", type=int, default=0,
                   help="Additional immutable checkpoints for external evaluation; 0 keeps the upstream interval")
    p.add_argument("--checkpoint-update-interval", type=int, default=0,
                   help="Save numbered checkpoints at exact policy-update milestones instead of control-step milestones")
    p.add_argument("--anchor-obs", action="store_true",
                   help="Use the upstream 9-D anchor-relative pose in environment and imagination")
    p.add_argument("--action-scale-mode", choices=("rl_lab", "torque_kp"), required=True)
    p.add_argument("--resume", type=Path)
    p.add_argument("--reference-initial", type=Path, help="Require exact initial network equality to this control checkpoint")
    p.add_argument("--save-initial-checkpoint", action="store_true",
                   help="Save the untrained/restored state for independent baseline evaluation")
    p.add_argument("--matmul-precision", choices=("ieee", "tf32"), default="ieee",
                   help="IEEE FP32 is the validated recomputation path; TF32 remains diagnostic")
    p.add_argument("--recompute-mode", choices=("all", "step", "policy"), default="all",
                   help="Recompute both paths, just the imagined step, or just the policy")
    p.add_argument("--config-only", action="store_true")
    args = p.parse_args()
    if args.resume and args.reference_initial:
        p.error("reference-initial is for fresh runs only")
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("run-name must be a single directory name")
    if min(args.stop_after_updates, args.extra_checkpoint_interval, args.checkpoint_update_interval) < 0:
        p.error("Update limit and extra checkpoint interval must be nonnegative")
    if args.stop_at_iteration is not None:
        if args.stop_at_iteration < 1:
            p.error("stop-at-iteration must be positive")
        args.stop_after_updates = 0
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        p.error("This validated memory path currently supports a single GPU")
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    lock = verify_sources()
    source = select_source(args.robot)
    from scripts.t800_action_scaling import install, checkpoint_metadata, assert_runtime_scale
    action_spec = install(args.action_scale_mode)
    import torch
    torch.set_num_threads(4)
    from supermimic.config import TrainConfig
    cfg = TrainConfig(run_name=args.run_name, resume=args.resume)
    cfg.data.npz_dir = args.motion_dir
    cfg.data.clips = args.clips
    cfg.env.anchor_obs = args.anchor_obs
    cfg.policy.goal_dim = cfg.env.future_frames * (68 if args.robot == "g1" else 60) + (9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    execution = {"mode": "activation_recompute", "robot": args.robot,
                 "action_parameterization": action_spec,
                 "baseline_commit": lock["baseline_commit"], "source_root": str(source),
                 "source_lock_sha256": sha(ROOT / "configs/strict-source-lock.json"),
                 "implementation_sha256": {str(path.relative_to(ROOT)): sha(path)
                                           for path in sorted((ROOT / "optimizations").glob("*.py"))},
                 "launcher_sha256": sha(Path(__file__)), "torch": str(torch.__version__),
                 "effective_policy_batch": cfg.policy_trainer.batch_size,
                 "horizon": cfg.policy_trainer.horizon, "dtype": "float32",
                 "backend_precision": args.matmul_precision,
                 "recompute_mode": args.recompute_mode,
                 "run_control": {"stop_after_updates": args.stop_after_updates,
                                 "stop_at_iteration": args.stop_at_iteration,
                                 "extra_checkpoint_interval": args.extra_checkpoint_interval,
                                 "checkpoint_update_interval": args.checkpoint_update_interval,
                                 "save_initial_checkpoint": args.save_initial_checkpoint}}
    checkpoint_hashes = {}
    if args.resume:
        with torch.serialization.safe_globals([Path, type(Path("."))]):
            ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        if checkpoint_metadata(ck) != action_spec:
            raise ValueError("Resume action scale differs: start a fresh run for this intervention")
        expected = json.loads(json.dumps(asdict(cfg), default=str))
        actual = json.loads(json.dumps(ck["config"], default=str))
        keys = ("env", "world_model", "policy", "critic", "wm_trainer", "policy_trainer", "schedule")
        changed = [k for k in keys if actual.get(k) != expected[k]]
        if changed or ck["config"].get("robot", "g1") != args.robot:
            raise ValueError(f"Resume recipe/robot mismatch: {changed}")
        if ck.get("execution", {}).get("mode") != "activation_recompute":
            raise ValueError("Resume requires an explicitly tagged memory-path checkpoint")
        if ck["execution"].get("backend_precision", "tf32") != args.matmul_precision:
            raise ValueError("Resume backend precision differs; use a fresh run for a precision change")
        if ck["execution"].get("recompute_mode", "all") != args.recompute_mode:
            raise ValueError("Resume recomputation mode differs; use the same execution configuration")
        for key in ("baseline_commit", "source_lock_sha256", "implementation_sha256"):
            if ck["execution"].get(key) != execution[key]:
                raise ValueError(f"Resume source provenance changed: {key}")
        for key in ("wm", "policy", "critic", "wm_opt", "policy_opt", "critic_opt"):
            checkpoint_hashes[key] = tensor_digest(ck[key])
        del ck
    log_dir = Path(cfg.log_root) / cfg.run_name
    destination = ROOT / "reports/microbatch" / args.run_name
    if not args.config_only and log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Use a fresh run-name; preserving existing logs: {log_dir}")
    destination.mkdir(parents=True, exist_ok=True)
    write_json(destination / "config.json", asdict(cfg))
    write_json(destination / "execution.json", execution)
    if args.config_only:
        print(json.dumps(execution, indent=2))
        return
    from optimizations.recompute import RecomputePolicyTrainer
    import supermimic.training.loop as loop
    # Selected only inside this new process; original source and strict entry stay untouched.
    loop.PolicyTrainer = RecomputePolicyTrainer
    if args.recompute_mode != "all":
        from functools import partial
        from optimizations.selective_recompute import SelectiveRecomputePolicyTrainer
        loop.PolicyTrainer = partial(SelectiveRecomputePolicyTrainer, recompute_mode=args.recompute_mode)
    class UpdateLimitReached(Exception):
        pass
    class MemoryTrainer(loop.Trainer):
        def act(self, obs):
            if args.checkpoint_update_interval and self.num_updates > initial_updates:
                if self.num_updates % args.checkpoint_update_interval == 0:
                    name = f"ckpt_u{self.num_updates:07d}.pt"
                    if not (self.log_dir / name).exists():
                        self.save(name)
            if args.extra_checkpoint_interval and self.iteration > self.start_iteration:
                if self.iteration % args.extra_checkpoint_interval == 0:
                    name = f"ckpt_{self.iteration:07d}.pt"
                    if not (self.log_dir / name).exists():
                        self.save(name)
            if args.stop_at_iteration is not None and self.iteration >= args.stop_at_iteration:
                raise UpdateLimitReached
            if args.stop_after_updates and self.num_updates >= self.limit_update:
                raise UpdateLimitReached
            return super().act(obs)

        def save(self, name="ckpt_latest.pt"):
            if args.checkpoint_update_interval and name.startswith("ckpt_") and name[5:12].isdigit():
                return  # Keep the original learning schedule; replace only checkpoint file cadence.
            if name == "ckpt_latest.pt" and getattr(self, "_last_numbered_state", None) == (self.iteration, self.num_updates):
                # Same immutable bytes, two names. Atomic replacement preserves the old milestone.
                temporary = self.log_dir / f".{name}.{os.getpid()}.link"
                os.link(self._last_numbered_path, temporary)
                temporary.replace(self.log_dir / name)
                return
            # Publish only after the complete checkpoint and provenance are written.
            temporary_name = f".{name}.{os.getpid()}.saving"
            temporary = self.log_dir / temporary_name
            super().save(temporary_name)
            with torch.serialization.safe_globals([Path, type(Path("."))]):
                saved = torch.load(temporary, map_location="cpu", weights_only=True)
            saved["execution"] = execution
            torch.save(saved, temporary)
            temporary.replace(self.log_dir / name)
            if name.startswith("ckpt_u"):
                self._last_numbered_state = (self.iteration, self.num_updates)
                self._last_numbered_path = self.log_dir / name
            del saved
            print(f"[CHECKPOINT] iteration {self.iteration}: {self.log_dir / name}", flush=True)

    report = {"execution": execution, "status": "running", "phase": "initialization",
              "limits": ["Memory implementation, not byte-identical upstream execution",
                         "Full original batch shapes; no mixed precision or reduced effective batch/horizon/network",
                         "Upstream resume refills replay and does not restore RNG; no uninterrupted-run equivalence claimed"]}
    path = destination / "result.json"
    write_json(path, report)
    trainer = None
    monitor = None
    monitor_log = None
    failed = False
    start_time = time.monotonic()
    try:
        try:
            monitor_log = (destination / "gpu-memory-samples.csv").open("w")
            monitor = subprocess.Popen(
                ["nvidia-smi", "--id=0", "--query-gpu=timestamp,memory.used,memory.total",
                 "--format=csv,noheader,nounits", "--loop-ms=250"],
                stdout=monitor_log, stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            report["device_memory_sampling_error"] = str(exc)
        torch.cuda.reset_peak_memory_stats()
        trainer = MemoryTrainer(cfg)
        assert_runtime_scale(trainer.env, action_spec)
        # make_env enables TF32. Override explicitly before the first forward/update.
        # The upstream source remains untouched; this execution difference is recorded.
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=args.matmul_precision == "tf32")
        execution["matmul_precision"] = torch.backends.cuda.matmul.fp32_precision
        execution["cudnn_precision"] = torch.backends.cudnn.conv.fp32_precision
        write_json(destination / "execution.json", execution)
        if args.stop_at_iteration is not None and args.stop_at_iteration <= trainer.iteration:
            raise ValueError("stop-at-iteration must exceed the resumed iteration")
        trainer.limit_update = trainer.num_updates + args.stop_after_updates
        initial_updates = trainer.num_updates
        report.update(start_iteration=trainer.iteration, start_updates=trainer.num_updates)
        if args.reference_initial:
            initial_control = torch.load(args.reference_initial, map_location="cpu", weights_only=False)
            checks = {key: tensor_digest(getattr(trainer, key).state_dict()) == tensor_digest(initial_control[key])
                      for key in ("wm", "policy", "critic")}
            if not all(checks.values()):
                raise ValueError(f"Initial networks differ from control: {checks}")
            report["initial_network_equality"] = checks
            report["reference_initial_checkpoint"] = str(args.reference_initial)
            del initial_control
        write_json(trainer.log_dir / "execution.json", execution)
        if checkpoint_hashes:
            restored = {"wm": trainer.wm.state_dict(), "policy": trainer.policy.state_dict(),
                        "critic": trainer.critic.state_dict(), "wm_opt": trainer.wm_trainer.opt.state_dict(),
                        "policy_opt": trainer.policy_trainer.opt.state_dict(),
                        "critic_opt": trainer.policy_trainer.critic_opt.state_dict()}
            checks = {key: tensor_digest(value) == checkpoint_hashes[key] for key, value in restored.items()}
            assert all(checks.values()), checks
            report["resume_exact_state_checks"] = checks
            del restored
        original_update = trainer.policy_trainer.update
        original_wm_update = trainer.wm_trainer.update
        last_wm = {}
        def measured_wm_update(buffer, *a, **kw):
            report.update(phase="world_model_update", iteration=trainer.iteration, updates=trainer.num_updates)
            write_json(path, report)
            t0 = time.monotonic()
            metrics = original_wm_update(buffer, *a, **kw)
            torch.cuda.synchronize()
            last_wm.update(metrics=metrics, seconds=time.monotonic() - t0)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Nonfinite world model metrics")
            return metrics
        def measured_update(buffer, *a, **kw):
            report.update(phase="policy_update", iteration=trainer.iteration, updates=trainer.num_updates)
            write_json(path, report)
            t0 = time.monotonic()
            metrics = original_update(buffer, *a, **kw)
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            row = {"update": trainer.num_updates + 1, "iteration": trainer.iteration,
                   "policy_seconds": time.monotonic() - t0,
                   "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
                   "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                   "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
                   "device_free_gib": free / 2**30, "device_used_gib": (total - free) / 2**30,
                   "metrics": metrics, "world_model_metrics": last_wm.get("metrics"),
                   "world_model_seconds": last_wm.get("seconds")}
            with (destination / "updates.jsonl").open("a") as out:
                out.write(json.dumps(row) + "\n")
            print(f"[MEMORY] update {row['update']} it {row['iteration']} | {row['policy_seconds']:.2f}s | "
                  f"torch peak {row['torch_peak_allocated_gib']:.2f} GiB | device free {row['device_free_gib']:.2f} GiB", flush=True)
            if not all(math.isfinite(v) for v in metrics.values()):
                raise FloatingPointError("Nonfinite update metrics; see updates.jsonl. No successful checkpoint saved.")
            return metrics
        trainer.policy_trainer.update = measured_update
        trainer.wm_trainer.update = measured_wm_update
        if args.save_initial_checkpoint:
            trainer.save("ckpt_initial.pt")
        report["phase"] = "upstream_training_loop"
        write_json(path, report)
        try:
            trainer.run()
        except UpdateLimitReached:
            trainer.save()
            if trainer.writer:
                trainer.writer.flush()
        torch.cuda.synchronize()
        report.update(status="passed", phase="saved", iteration=trainer.iteration,
                      updates=trainer.num_updates, additional_updates=trainer.num_updates - initial_updates,
                      checkpoint=str(trainer.log_dir / "ckpt_latest.pt"),
                      checkpoint_sha256=sha(trainer.log_dir / "ckpt_latest.pt"))
    except KeyboardInterrupt:
        failed = True
        report.update(status="interrupted", error="Interrupted; retain only previously completed checkpoints")
    except Exception as exc:
        failed = True
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - start_time
        report["torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if trainer is not None:
            trainer.env.close()
            if trainer.writer:
                trainer.writer.close()
        if monitor is not None:
            monitor.terminate()
            monitor.wait(timeout=5)
        if monitor_log is not None:
            monitor_log.close()
            samples = []
            for line in (destination / "gpu-memory-samples.csv").read_text().splitlines():
                fields = line.split(",")
                if len(fields) == 3:
                    try:
                        samples.append(float(fields[1]))
                    except ValueError:
                        pass
            if samples:
                report["device_sampled_peak_used_gib"] = max(samples) / 1024
                report["device_memory_sampling"] = "nvidia-smi, 250 ms; whole GPU including desktop and other processes; may miss shorter peaks"
        write_json(path, report)
    print(json.dumps({k: report.get(k) for k in ("status", "iteration", "updates", "torch_peak_allocated_gib", "elapsed_seconds")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
