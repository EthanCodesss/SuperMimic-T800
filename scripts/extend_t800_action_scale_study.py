"""Explicit exploratory continuation after a positive but sub-threshold screen.

The original gate remains failed and unchanged. This is not a preregistered gate
pass: the reason for extending is recorded before any additional training.
"""
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_t800_action_scale_study import (
    ROOT, OUT, RUN2000, RUN8000, BASE8000, write, sha, train_stage, evaluate_control,
)
import json


def main():
    os.chdir(ROOT)
    os.environ.update(SUPERMIMIC_ROBOT="t800", MPLCONFIGDIR=str(ROOT/".cache/matplotlib"),
                      OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    gate=json.loads((OUT/"gate-result.json").read_text())
    status=json.loads((OUT/"status.json").read_text())
    assert status["status"]=="complete" and not gate["extend"]
    assert gate["treatment"]["joint_position_rmse_rad"] < gate["control"]["joint_position_rmse_rad"]
    assert gate["checks"]["full_clip_return"] and gate["checks"]["reset_completion"]
    decision=OUT/"exploratory-extension.json"
    if decision.exists():
        raise FileExistsError(decision)
    for name in ("delivery.json","report.html","joint-comparison.png","training-curves.png","status.json"):
        shutil.copy2(OUT/name, OUT/("screening-2000-"+name))
    write(decision, dict(created_utc=datetime.now(timezone.utc).isoformat(),
        classification="post-screen exploratory extension; original threshold not met",
        original_gate_passed=False, original_gate_sha256=sha(OUT/"gate-result.json"),
        reason="The screen improved joint RMSE by 8.5%, return and all five perturbed rollouts, but missed the 10% gate. A same-budget 8000 comparison is needed to distinguish slower convergence from final capability. The gate is not retrospectively changed.",
        additional_policy_updates=6000, maximum_total_policy_updates=8000,
        resume_checkpoint=f"logs/{RUN2000}/ckpt_latest.pt",
        resume_checkpoint_sha256=sha(ROOT/f"logs/{RUN2000}/ckpt_latest.pt"),
        launcher_sha256=sha(Path(__file__))))
    status.update(status="running",active_stage=RUN8000,pid=os.getpid(),exploratory_extension=True)
    status.pop("conclusion",None)
    write(OUT/"status.json",status)
    try:
        train_stage(RUN8000,6000,f"logs/{RUN2000}/ckpt_latest.pt")
        status["completed"].append(RUN8000)
        status.update(active_stage="control_8000_evaluation")
        write(OUT/"status.json",status)
        evaluate_control(BASE8000,8000)
        status.update(status="evaluations_complete",active_stage=None,finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT/"status.json",status)
    except BaseException as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        write(OUT/"status.json",status)
        raise


if __name__=="__main__":
    main()
