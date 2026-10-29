"""Run the corrected G1 experiment with exact initial-state checks and evaluations.

The training recipe is unchanged. Evaluations run in separate processes and
consume no training RNG; extra checkpoints are atomically published. This
driver can remain running while status/evaluations are read from reports/.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, default=str) + "\n")
    temporary.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("g1",), required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--run-name", required=True)
    stop = p.add_mutually_exclusive_group(required=True)
    stop.add_argument("--stop-at-iteration", type=int)
    stop.add_argument("--stop-after-updates", type=int)
    p.add_argument("--checkpoint-interval", type=int, default=1000)
    p.add_argument("--checkpoint-update-interval", type=int, default=0)
    p.add_argument("--checkpoint-updates", type=int, nargs="+", default=[2000, 4052, 8104])
    p.add_argument("--anchor-obs", action="store_true")
    p.add_argument("--wm-test-windows", type=Path,
                   help="Reuse fixed model-test transitions across stages and observation variants")
    p.add_argument("--robustness-seeds", type=int, nargs="*", default=[])
    p.add_argument("--defer-evaluation", action="store_true",
                   help="Evaluate saved milestones after training exits so they have exclusive GPU compute")
    p.add_argument("--resume", type=Path)
    p.add_argument("--reference-initial", type=Path, required=True)
    p.add_argument("--matmul-precision", choices=("ieee", "tf32"), default="ieee")
    p.add_argument("--recompute-mode", choices=("all", "step", "policy"), default="all")
    p.add_argument("--protocol", type=Path, required=True, help="Frozen run specification, checked before launching")
    args = p.parse_args()
    os.chdir(ROOT)
    if Path(args.run_name).name != args.run_name or args.run_name in (".", ".."):
        p.error("Use a single directory name")
    if args.checkpoint_interval < 1 or (args.stop_at_iteration or args.stop_after_updates or 0) < 1:
        p.error("Intervals must be positive")
    if not args.motion.is_file():
        p.error("Motion must be an existing NPZ file")
    if args.protocol:
        import hashlib
        protocol = json.loads(args.protocol.read_text())
        assert protocol["run_name"] == args.run_name
        assert protocol["robot"] == args.robot
        assert Path(protocol["motion"]["path"]).resolve() == args.motion.resolve()
        assert protocol["budget"]["additional_updates"] == args.stop_after_updates
        assert protocol["evaluation"]["checkpoint_interval_control_steps"] == args.checkpoint_interval
        assert protocol.get("anchor_obs", False) == args.anchor_obs
        assert protocol["evaluation"].get("checkpoint_interval_updates", 0) == args.checkpoint_update_interval
        assert protocol["evaluation"]["checkpoint_updates"] == sorted(set(args.checkpoint_updates))
        assert Path(protocol["reference_initial"]).resolve() == args.reference_initial.resolve()
        assert (protocol["initialization"] == "random_seed_0") == (args.resume is None)
        assert protocol["execution_precision"]["matmul"] == args.matmul_precision
        assert protocol["execution_optimization"]["recompute_mode"] == args.recompute_mode
        assert protocol["evaluation"]["robustness_seeds_final"] == args.robustness_seeds
        assert Path(protocol["evaluation"]["fixed_wm_windows"]).resolve() == args.wm_test_windows.resolve()
        for name, expected in protocol["frozen_file_sha256"].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    destination = ROOT / "reports/training" / args.run_name
    training_dir = ROOT / "logs" / args.run_name
    if destination.exists() or training_dir.exists():
        raise FileExistsError("Use a fresh run-name to preserve existing training/evaluation records")
    # Reserve for numbered checkpoints, latest checkpoint, and an in-flight save.
    # Original recipe: 0.4 updates/control step and 40 steps of replay warm-up.
    upper_steps = args.stop_at_iteration or (int(args.stop_after_updates / .4) + 42)
    reference = args.resume or ROOT / f"logs/{args.robot}_memory_verified/ckpt_latest.pt"
    checkpoint_bytes = max(reference.stat().st_size if reference.exists() else 0, 256 * 1024**2)
    estimated = (upper_steps // args.checkpoint_interval + upper_steps // 4000 + 5) * checkpoint_bytes
    if args.checkpoint_update_interval:
        estimated = ((args.stop_after_updates or int(upper_steps * .4)) // args.checkpoint_update_interval + 3) * checkpoint_bytes
    if args.checkpoint_updates and not args.checkpoint_update_interval:
        estimated = (len(set(args.checkpoint_updates)) + 3) * checkpoint_bytes
    free = shutil.disk_usage(ROOT).free
    if estimated > free - 1024**3:
        raise OSError(f"Stage checkpoints need approximately {estimated / 2**30:.1f} GiB; only {free / 2**30:.1f} GiB free")
    destination.mkdir(parents=True)
    env = os.environ.copy()
    env.update(MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"), OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    status = {"run_name": args.run_name, "robot": args.robot, "stage_pid": os.getpid(), "status": "preparing",
              "target_iteration": args.stop_at_iteration, "checkpoint_interval": args.checkpoint_interval,
              "additional_update_budget": args.stop_after_updates,
              "matmul_precision": args.matmul_precision,
              "recompute_mode": args.recompute_mode,
              "anchor_obs": args.anchor_obs, "checkpoint_interval_updates": args.checkpoint_update_interval,
              "checkpoint_updates": args.checkpoint_updates, "reference_initial": str(args.reference_initial),
              "fk_compile_compat": protocol["fk_compile_compat"],
              "fixed_wm_windows": str(args.wm_test_windows) if args.wm_test_windows else None,
              "robustness_seeds": args.robustness_seeds,
              "defer_evaluation": args.defer_evaluation,
              "resume": str(args.resume) if args.resume else None, "estimated_checkpoint_gib": estimated / 2**30,
              "protocol": str(args.protocol) if args.protocol else None,
              "motion": str(args.motion), "evaluations": [],
              "limits": ["Single deterministic evaluation per checkpoint, not a multi-seed success rate",
                         "Evaluation is separate from randomized ten-second training episodes",
                         "Extra checkpoint/stop controls do not change the 320k-step LR schedule"]}
    path = destination / "status.json"
    write_json(path, status)
    evaluations = []
    fixed_windows = args.wm_test_windows
    robustness = []

    def evaluate(checkpoint, label, seed=0, reset_noise=False):
        nonlocal fixed_windows
        output = destination / f"{label}.json"
        status["active_evaluation"] = str(checkpoint)
        write_json(path, status)
        cmd = [sys.executable, str(ROOT / "scripts/evaluate_fk_compat.py"), "--robot", args.robot,
               "--checkpoint", str(checkpoint), "--motion", status["motion"],
               "--wm-horizon", "32", "--out", str(output), "--seed", str(seed)]
        if reset_noise:
            cmd.append("--reset-noise")
        if fixed_windows is not None:
            cmd.extend(["--wm-test-windows", str(fixed_windows)])
        with output.with_suffix(".log").open("w") as log:
            run = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            status.setdefault("evaluation_errors", []).append({"checkpoint": str(checkpoint), "log": str(output.with_suffix('.log'))})
            print(f"[EVAL ERROR] {checkpoint}; inspect {output.with_suffix('.log')}", flush=True)
        else:
            status["evaluation_errors"] = [item for item in status.get("evaluation_errors", [])
                                           if item["checkpoint"] != str(checkpoint)]
            record = json.loads(output.read_text())
            if reset_noise:
                robustness.append(record)
                write_json(destination / "robustness.json", robustness)
            else:
                evaluations.append(record)
                write_json(destination / "evaluations.json", evaluations)
            if fixed_windows is None and label == "baseline" and output.with_suffix(".wm_windows.pt").exists():
                fixed_windows = output.with_suffix(".wm_windows.pt")
            summary = {k: record[k] for k in ("checkpoint_iteration", "checkpoint_updates", "steps", "completed_clip",
                       "completion_fraction", "normalized_return", "mean_root_position_error_m")}
            summary["report"] = str(output)
            if not reset_noise:
                status["evaluations"].append(summary)
            print(f"[EVAL] it {record['checkpoint_iteration']} | {record['steps'] / 50:.2f}s | "
                  f"return {record['normalized_return']:.4f} | complete {record['completed_clip']}", flush=True)
        status.pop("active_evaluation", None)
        write_json(path, status)
        if run.returncode:
            raise RuntimeError(f"Evaluation failed; inspect {output.with_suffix('.log')}")

    def evaluate_milestone(checkpoint, label):
        evaluate(checkpoint, label)
        output = destination / f"{label}.upstream.json"
        command = [sys.executable, str(ROOT / "scripts/evaluate_upstream_e1c.py"),
                   "--checkpoint", str(checkpoint), "--out", str(output)]
        with output.with_suffix(".log").open("w") as log:
            run = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if run.returncode:
            raise RuntimeError(f"Upstream benchmark failed: {output}")
        status.setdefault("upstream_benchmarks", []).append(str(output))
        write_json(path, status)

    if args.resume:
        evaluate(args.resume, "baseline")
    command = [sys.executable, "-u", str(ROOT / "scripts/train_memory_fk_compat.py"), "--robot", args.robot,
               "--motion-dir", str(args.motion.parent), "--clips", args.motion.stem,
               "--run-name", args.run_name,
               "--matmul-precision", args.matmul_precision,
               "--recompute-mode", args.recompute_mode,
               "--extra-checkpoint-interval", str(0 if args.checkpoint_update_interval or args.checkpoint_updates else args.checkpoint_interval),
               "--checkpoint-update-interval", str(args.checkpoint_update_interval),
               "--checkpoint-updates", *map(str, args.checkpoint_updates),
               "--reference-initial", str(args.reference_initial)]
    if args.anchor_obs:
        command.append("--anchor-obs")
    command += (["--stop-at-iteration", str(args.stop_at_iteration)] if args.stop_at_iteration
                else ["--stop-after-updates", str(args.stop_after_updates)])
    command += ["--resume", str(args.resume)] if args.resume else ["--save-initial-checkpoint"]
    start = time.monotonic()
    with (destination / "training.log").open("w") as training_log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=training_log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        status.update(status="training", training_pid=process.pid, training_command=command)
        write_json(path, status)
        seen = set()
        deferred = set()
        last_print = 0.
        try:
            while True:
                initial = training_dir / "ckpt_initial.pt"
                can_evaluate = not args.defer_evaluation or process.poll() is not None
                if can_evaluate and not args.resume and initial.exists() and initial.name not in seen and (initial.name not in deferred or process.poll() is not None):
                    try:
                        evaluate(initial, "baseline")
                        seen.add(initial.name)
                    except RuntimeError:
                        if process.poll() is not None:
                            raise
                        deferred.add(initial.name)
                        print("[EVAL DEFERRED] baseline; retry after training releases GPU", flush=True)
                numbered = list(training_dir.glob("ckpt_[0-9]*.pt")) + list(training_dir.glob("ckpt_u[0-9]*.pt"))
                pending = sorted(p for p in numbered if p.name not in seen and (p.name not in deferred or process.poll() is not None)) if can_evaluate else []
                for checkpoint in pending:
                    try:
                        evaluate_milestone(checkpoint, f"eval_{checkpoint.stem}")
                        seen.add(checkpoint.name)
                    except RuntimeError:
                        if process.poll() is not None:
                            raise
                        deferred.add(checkpoint.name)
                        print(f"[EVAL DEFERRED] {checkpoint.name}; retry after training releases GPU", flush=True)
                elapsed = time.monotonic() - start
                runtime_path = ROOT / "reports/microbatch" / args.run_name / "result.json"
                if runtime_path.exists():
                    try:
                        runtime = json.loads(runtime_path.read_text())
                        status["training_progress"] = {k: runtime.get(k) for k in ("status", "phase", "iteration", "updates", "error")}
                    except json.JSONDecodeError:
                        pass
                status["training_elapsed_seconds"] = elapsed
                write_json(path, status)
                code = process.poll()
                if code is not None:
                    # Re-scan after exit in case a checkpoint appeared during polling.
                    numbered = list(training_dir.glob("ckpt_[0-9]*.pt")) + list(training_dir.glob("ckpt_u[0-9]*.pt"))
                    if any(p.name not in seen for p in numbered):
                        continue
                    status["training_returncode"] = code
                    if code == 0:
                        final_updates = status["training_progress"]["updates"]
                        numbered_output = destination / f"eval_ckpt_u{final_updates:07d}.json"
                        if numbered_output.exists():
                            record = json.loads(numbered_output.read_text())
                            record["evaluation_reused_from"] = str(numbered_output)
                            write_json(destination / "final.json", record)
                        else:
                            evaluate(training_dir / "ckpt_latest.pt", "final")
                        # The final model normally has an already evaluated immutable update checkpoint.
                        final_updates = evaluations[-1]["checkpoint_updates"]
                        if not any(r["checkpoint_updates"] == final_updates for r in robustness):
                            for seed in args.robustness_seeds:
                                evaluate(training_dir / "ckpt_latest.pt", f"final_reset_seed_{seed}", seed=seed, reset_noise=True)
                    status["status"] = "finished" if code == 0 and not status.get("evaluation_errors") else "failed"
                    write_json(path, status)
                    subprocess.run([sys.executable, str(ROOT / "scripts/report_g1_fkfix_study.py")], cwd=ROOT, env=env)
                    print(f"[STAGE] {status['status']} after {elapsed / 3600:.2f} h", flush=True)
                    if status["status"] != "finished":
                        raise SystemExit(1)
                    break
                if elapsed - last_print >= 60:
                    last_print = elapsed
                    print(f"[PROGRESS] {elapsed / 3600:.2f} h | {status.get('training_progress', {})}", flush=True)
                    subprocess.run([sys.executable, str(ROOT / "scripts/report_g1_fkfix_study.py")], cwd=ROOT, env=env)
                time.sleep(2)
        except BaseException as exc:
            status.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                          error=f"{type(exc).__name__}: {exc}")
            write_json(path, status)
            raise
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()


if __name__ == "__main__":
    main()
