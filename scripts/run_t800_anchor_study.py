"""Run the user's four-stage T800 study, sequentially and with bounded budgets."""
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/t800_anchor_study"
SHORT = "data/npz/t800_single/walk2_steady.npz"
LONG = "data/npz/t800_long/walk2_continuous60.npz"
FIXED = "reports/training/t800_walk2_policy_ieee_stage1/baseline.wm_windows.pt"
BASE = "logs/t800_walk2_policy_ieee_stage1/ckpt_latest.pt"
STAGES = [
    dict(name="t800_short_noanchor_8000", motion=SHORT, anchor=False, updates=6000, resume=BASE),
    dict(name="t800_short_anchor_8000", motion=SHORT, anchor=True, updates=8000, resume=None),
    dict(name="t800_long_noanchor_8000", motion=LONG, anchor=False, updates=8000, resume=None),
    dict(name="t800_long_anchor_8000", motion=LONG, anchor=True, updates=8000, resume=None),
]


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def protocol(stage):
    import numpy as np
    p = copy.deepcopy(json.loads((ROOT / "configs/t800-single-motion-policy-ieee-baseline.json").read_text()))
    p.update(created_utc=datetime.now(timezone.utc).isoformat(), run_name=stage["name"],
             anchor_obs=stage["anchor"], initialization="resume_existing_2000" if stage["resume"] else "random_seed_0")
    with np.load(ROOT / stage["motion"]) as motion:
        p["motion"] = dict(path=stage["motion"], frames=len(motion["joint_pos"]), fps=float(motion["fps"][0]),
                           duration_s=(len(motion["joint_pos"])-1)/float(motion["fps"][0]))
    p["schema"]["goal_dim"] = 609 if stage["anchor"] else 600
    p["budget"]["additional_updates"] = stage["updates"]
    p["budget"]["target_total_updates"] = 8000
    p["evaluation"].update(checkpoint_interval_control_steps=1000, checkpoint_interval_updates=2000,
                            fixed_wm_windows=FIXED, robustness_seeds_each_milestone=[1, 2, 3, 4, 5],
                            defer_evaluation=True,
                            full_clip_return="All reference frames in denominator; unplayed rewards are zero")
    p["execution_precision"]["cudnn"] = "tf32 (actual setting; models do not use convolutions)"
    p["adapter_decisions"] += ["Anchor relative observation follows the upstream optional implementation" if stage["anchor"]
                              else "Default proprioceptive observation; no anchor feedback"]
    p["comparison_limits"] = ["One training seed; five reset-noise episodes are not five training seeds",
                               "Short no-anchor run resumes after 2000 updates, refilling replay; others start fresh",
                               "Both long runs start fresh and use the identical continuous motion and budget",
                               "Fixed WM windows come from the original short untrained rollout; on-policy WM covers the evaluated clip"]
    frozen = [ROOT / n for n in p["frozen_file_sha256"] if (ROOT / n).exists()]
    frozen += [ROOT / stage["motion"], ROOT / FIXED, Path(__file__)]
    p["frozen_file_sha256"] = {str(f.relative_to(ROOT)): sha(f) for f in frozen}
    if stage["resume"]:
        p["resume_checkpoint"] = dict(path=stage["resume"], sha256=sha(ROOT / stage["resume"]))
    destination = ROOT / "configs" / f"{stage['name']}.json"
    if destination.exists():
        raise FileExistsError(destination)
    write(destination, p)
    return destination


def prepare_long():
    if not (ROOT / LONG).exists():
        run([sys.executable, "scripts/prepare_t800.py", "--bvh", "data/raw/lafan1/walk2_subject1.bvh",
             "--start", "300", "--frames", "1801", "--out", LONG], OUT / "prepare-long.log")
    run([sys.executable, "scripts/audit_motion.py", LONG, "--out", str(OUT / "long-motion-audit.json")],
        OUT / "long-motion-audit.log")
    audit = json.loads((OUT / "long-motion-audit.json").read_text())
    if not audit["fk"]["passed"] or not all(m["passed"] for m in audit["motions"]):
        raise RuntimeError("Long reference failed feasibility audit; inspect before training")


def main():
    os.chdir(ROOT)
    os.environ.update(SUPERMIMIC_ROBOT="t800", MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
                      OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    OUT.mkdir(parents=True, exist_ok=True)
    validation = json.loads((OUT / "validation/step1.json").read_text())
    if not validation["passed"]:
        raise RuntimeError("Evaluation and anchor launcher validation must pass first")
    state = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                 stages=STAGES, completed=[], validation=validation)
    write(OUT / "status.json", state)
    try:
        for i, stage in enumerate(STAGES):
            state.update(stage_index=i, active_stage=stage["name"])
            write(OUT / "status.json", state)
            if i == 2:
                prepare_long()
            spec = protocol(stage)
            cmd = [sys.executable, "scripts/run_training_stage.py", "--robot", "t800",
                   "--motion", stage["motion"], "--run-name", stage["name"],
                   "--stop-after-updates", str(stage["updates"]), "--checkpoint-update-interval", "2000",
                   "--matmul-precision", "ieee", "--recompute-mode", "policy", "--protocol", str(spec),
                   "--defer-evaluation", "--wm-test-windows", FIXED, "--robustness-seeds", "1", "2", "3", "4", "5"]
            if stage["anchor"]:
                cmd.append("--anchor-obs")
            if stage["resume"]:
                cmd += ["--resume", stage["resume"]]
            run(cmd, OUT / f"{stage['name']}-driver.log")
            final = json.loads((ROOT / f"reports/training/{stage['name']}/final.json").read_text())
            if final["checkpoint_updates"] != 8000:
                raise RuntimeError(f"Expected exactly 8000 updates: {final['checkpoint_updates']}")
            state["completed"].append(stage["name"])
            write(OUT / "status.json", state)
        state.update(status="training_completed", finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / "status.json", state)
        run([sys.executable, "scripts/report_t800_anchor_study.py"], OUT / "report-generation.log")
        state["status"] = "complete"
        write(OUT / "status.json", state)
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(OUT / "status.json", state)
        raise


if __name__ == "__main__":
    main()
