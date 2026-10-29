"""Compare activation-storage choices with the same full T800 training recipe."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision", choices=("ieee", "tf32"), default="ieee")
    p.add_argument("--modes", nargs="+", choices=("all", "step", "policy"), default=["all", "step", "policy"])
    p.add_argument("--updates", type=int, default=24)
    p.add_argument("--prefix", default="t800_storage_bench")
    args = p.parse_args()
    if args.updates < 12:
        p.error("Use at least 12 updates to measure beyond initialization")
    os.chdir(ROOT)
    output = ROOT / "reports" / f"{args.prefix}-{args.precision}.json"
    if output.exists():
        raise FileExistsError(output)
    results = []
    for mode in args.modes:
        run_name = f"{args.prefix}_{mode}_{args.precision}"
        command = [sys.executable, "-u", "scripts/train_memory.py", "--robot", "t800",
                   "--motion-dir", "data/npz/t800_single", "--clips", "walk2_steady",
                   "--run-name", run_name, "--stop-after-updates", str(args.updates),
                   "--matmul-precision", args.precision, "--recompute-mode", mode]
        print(f"[BENCH] {mode}, {args.precision}, {args.updates} updates", flush=True)
        log_path = ROOT / "reports" / f"{run_name}.log"
        with log_path.open("x") as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        report_dir = ROOT / "reports/microbatch" / run_name
        record = json.loads((report_dir / "result.json").read_text()) if (report_dir / "result.json").exists() else {}
        rows = [json.loads(s) for s in (report_dir / "updates.jsonl").read_text().splitlines()] if (report_dir / "updates.jsonl").exists() else []
        stable = rows[8:]
        result = {"mode": mode, "precision": args.precision, "returncode": completed.returncode,
                  "run_name": run_name, "status": record.get("status"), "error": record.get("error"),
                  "updates": len(rows), "torch_peak_allocated_gib": record.get("torch_peak_allocated_gib"),
                  "device_sampled_peak_used_gib": record.get("device_sampled_peak_used_gib"),
                  "median_policy_seconds": statistics.median(r["policy_seconds"] for r in stable) if stable else None,
                  "median_update_seconds": statistics.median(r["policy_seconds"] + r["world_model_seconds"] for r in stable) if stable else None,
                  "log": str(log_path.relative_to(ROOT))}
        results.append(result)
        output.write_text(json.dumps({"results": results, "scope": "24 full updates per mode; excludes first 8 updates for timing; not convergence"}, indent=2) + "\n")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
