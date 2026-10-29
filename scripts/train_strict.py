"""Pinned upstream recipe, with only T800 body/schema substitutions when selected.

G1 imports the pristine vendor source. No network/batch/noise/schedule overrides
are exposed. Probe performs the first scheduled update and writes no checkpoint.
The CUDA 12 compatibility environment still differs from upstream's CUDA 13 one.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def verify_sources():
    lock = json.loads((ROOT / "configs/strict-source-lock.json").read_text())
    for directory, expected in lock["source_trees"].items():
        actual = {str(p.relative_to(ROOT / directory)): sha(p)
                  for p in sorted((ROOT / directory).rglob("*.py"))}
        if actual != expected:
            changed = sorted(k for k in actual.keys() | expected.keys()
                             if actual.get(k) != expected.get(k))
            raise RuntimeError(f"Source differs from audited lock: {directory}: {changed}")
    return lock


def select_source(robot):
    # Set before ANY supermimic import, including the editable installation.
    os.environ["SUPERMIMIC_ROBOT"] = robot
    source = ROOT / ("vendor/supermimic_upstream/src" if robot == "g1" else "src")
    sys.path.insert(0, str(source))
    import supermimic.config
    assert Path(supermimic.config.__file__).resolve().is_relative_to(source)
    return source


def capacity_probe(trainer, report, destination):
    """Same env steps and first update as Trainer.run, without a long run/save.

    The replay is allocated at full capacity. This only measures the first
    update's capacity; later compilation/allocator growth may require more.
    """
    import torch
    from supermimic.training.schedule import cosine_lr

    def phase(name):
        report["phase"] = name
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        report.setdefault("memory", []).append({
            "phase": name, "device_free_gib": free / 2**30,
            "device_total_gib": total / 2**30,
            "torch_allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "torch_peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "torch_reserved_gib": torch.cuda.memory_reserved() / 2**30,
        })
        write_json(destination, report)
        print(f"[PROBE] {name}; device free {free / 2**30:.2f} GiB", flush=True)

    cfg = trainer.cfg
    sch = cfg.schedule
    phase("reset")
    obs, _ = trainer.env.reset()
    phase("collect_first_update_data")
    while True:
        action = trainer.act(obs)
        obs, rew, term, tout, _ = trainer.env.step(action)
        trainer.ep_reward_sum += rew
        trainer.ep_len += 1
        done = term | tout
        if done.any():
            trainer.ep_reward_sum[done] = 0.
            trainer.ep_len[done] = 0.
        torch.cuda.synchronize()
        trainer.iteration += 1
        if trainer.iteration - trainer.start_iteration >= sch.warmup_iterations:
            trainer._utd_acc += sch.updates_per_env_step
        if trainer._utd_acc >= 1.:
            trainer._utd_acc -= 1.
            break
    lr = cosine_lr(trainer.iteration, sch.total_iterations, cfg.wm_trainer.lr,
                   sch.lr_final_ratio, sch.lr_decay_fraction)
    trainer.wm_trainer.set_lr(lr)
    trainer.policy_trainer.set_lr(lr * cfg.policy_trainer.lr / cfg.wm_trainer.lr)
    report.update(iteration=trainer.iteration, lr=lr, replay_rows=trainer.buffer.size)
    phase("world_model_update")
    report["world_model_metrics"] = trainer.wm_trainer.update(trainer.buffer)
    phase("policy_and_critic_update")
    for _ in range(sch.policy_updates_per_wm_update):
        report["policy_metrics"] = trainer.policy_trainer.update(trainer.buffer)
    trainer.num_updates += 1
    phase("first_update_complete")
    report["status"] = "passed_first_update_only"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("g1", "t800"), required=True)
    p.add_argument("--mode", choices=("config", "probe", "train"), default="config")
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--clips", nargs="+")
    p.add_argument("--run-name", required=True)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("run-name must be a single directory name")
    if args.mode == "probe" and args.resume:
        p.error("capacity probe starts from the upstream initialization, without resume")
    os.chdir(ROOT)
    lock = verify_sources()
    source = select_source(args.robot)
    from supermimic.config import TrainConfig
    cfg = TrainConfig(run_name=args.run_name, resume=args.resume)
    cfg.data.npz_dir = args.motion_dir
    cfg.data.clips = args.clips
    # These are the same derived fields that Trainer.__init__ resolves.
    feature_dim = 68 if args.robot == "g1" else 60
    cfg.policy.goal_dim = cfg.env.future_frames * feature_dim + (9 if cfg.env.anchor_obs else 0)
    cfg.policy_trainer.anchor_obs = cfg.env.anchor_obs
    if args.resume:
        import torch
        with torch.serialization.safe_globals([Path, type(Path("."))]):
            ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        if ck.get("execution", {}).get("mode") in ("trajectory_microbatch", "activation_recompute"):
            raise ValueError("Checkpoint was produced by the memory implementation; resume with scripts/train_memory.py")
        if ck["config"].get("robot", "g1") != args.robot:
            raise ValueError("Checkpoint robot does not match")
        expected = json.loads(json.dumps(asdict(cfg), default=str))
        actual = json.loads(json.dumps(ck["config"], default=str))
        keys = ("env", "world_model", "policy", "critic", "wm_trainer", "policy_trainer", "schedule")
        changed = [k for k in keys if actual.get(k) != expected[k]]
        if changed:
            raise ValueError(f"Checkpoint uses a different recipe: {changed}; cannot resume as strict")
        del ck
    destination = ROOT / "reports/strict" / args.run_name
    write_json(destination / "config.json", asdict(cfg))
    import torch
    import mujoco
    import warp
    report = {
        "robot": args.robot, "mode": args.mode, "baseline_commit": lock["baseline_commit"],
        "source_root": str(source), "source_lock_sha256": sha(ROOT / "configs/strict-source-lock.json"),
        "launcher_sha256": sha(Path(__file__)),
        "versions": {"python": sys.version, "torch": torch.__version__,
                     "mujoco": mujoco.__version__, "warp": warp.__version__},
        "limits": ["Source/recipe alignment, not full dependency/binary parity",
                   "T800 requires body/schema/control substitutions",
                   "Probe only covers allocation, collection and the first complete update; not convergence"],
        "status": "config_verified", "phase": "config",
    }
    path = destination / "result.json"
    write_json(path, report)
    print(f"[STRICT] {args.robot}: {source}; {args.mode}; config {destination / 'config.json'}", flush=True)
    if args.mode == "config":
        return
    log_dir = Path(cfg.log_root) / cfg.run_name
    if log_dir.exists() and any(log_dir.iterdir()):
        raise FileExistsError(f"Use a fresh run-name; preserving existing logs: {log_dir}")
    from supermimic.training.loop import Trainer
    trainer = None
    start = time.monotonic()
    failed = False
    try:
        report.update(status="running", phase="trainer_initialization")
        write_json(path, report)
        torch.cuda.reset_peak_memory_stats()
        trainer = Trainer(cfg)
        if args.mode == "probe":
            capacity_probe(trainer, report, path)
        else:
            report["phase"] = "upstream_training_loop"
            write_json(path, report)
            trainer.run()
            report["status"] = "training_loop_finished"
    except Exception as exc:
        failed = True
        message = str(exc)
        report.update(status="cuda_out_of_memory" if isinstance(exc, torch.OutOfMemoryError)
                      or "out of memory" in message.lower() else "failed",
                      error_type=type(exc).__name__, error=message)
        traceback.print_exc()
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        report["torch_peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
        write_json(path, report)
        if trainer is not None:
            trainer.env.close()
            if trainer.writer is not None:
                trainer.writer.close()
    print(json.dumps({k: report[k] for k in ("status", "phase", "elapsed_seconds", "torch_peak_allocated_gib")}), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
