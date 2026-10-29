"""Freeze the audited T800 single-motion protocol without running training."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_strict import sha, verify_sources


def differences(a, b, prefix=""):
    if isinstance(a, dict) and isinstance(b, dict):
        result = {}
        for k in sorted(a.keys() | b.keys()):
            result.update(differences(a.get(k), b.get(k), f"{prefix}.{k}".strip(".")))
        return result
    return {} if a == b else {prefix: {"g1": a, "t800": b}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name")
    p.add_argument("--out", type=Path)
    p.add_argument("--matmul-precision", choices=("ieee", "tf32"), default="ieee")
    p.add_argument("--recompute-mode", choices=("all", "step", "policy"), required=True)
    p.add_argument("--parity-report", type=Path, required=True)
    args = p.parse_args()
    args.run_name = args.run_name or f"t800_walk2_{args.recompute_mode}_{args.matmul_precision}_stage1"
    args.out = args.out or Path(f"configs/t800-single-motion-{args.recompute_mode}-{args.matmul_precision}-baseline.json")
    os.chdir(ROOT)
    if args.out.exists():
        raise FileExistsError("Preserve the frozen protocol; choose a new output")
    lock = verify_sources()
    cfgs = {}
    for robot in ("g1", "t800"):
        code = """
import json, sys
from dataclasses import asdict
from scripts.train_strict import select_source
select_source(sys.argv[1])
from supermimic.config import TrainConfig
print(json.dumps(asdict(TrainConfig()), default=str))
"""
        cfgs[robot] = json.loads(subprocess.check_output([sys.executable, "-c", code, robot], text=True))
    diff = differences(cfgs["g1"], cfgs["t800"])
    allowed = {"robot", "policy.goal_dim", "world_model.state_spec.xi_offset", "world_model.state_spec.xi_scale"}
    assert set(diff) == allowed, diff
    import numpy as np
    from scripts.train_strict import select_source
    select_source("t800")
    from supermimic.robots.spec import NJ, NQ, NV, STATE_DIM, DELTA_DIM, OBS_FRAME_DIM, GOAL_FEAT_DIM
    from supermimic.robots.t800 import JOINT_NAMES, TRACKED_BODIES, XML, load_t800_model
    model = load_t800_model()
    assert NJ == len(JOINT_NAMES) == 25 and (model.nq, model.nv) == (NQ, NV)
    motion_path = Path("data/npz/t800_single/walk2_steady.npz")
    with np.load(motion_path) as data:
        assert tuple(data["joint_names"].tolist()) == JOINT_NAMES
        assert data["joint_pos"].shape[1] == NJ
        assert set(TRACKED_BODIES).issubset(set(data["body_names"].tolist()))
        assert np.isfinite(data["joint_pos"]).all()
        head = data["joint_pos"][:, 23:25]
        motion = {"path": str(motion_path), "frames": len(data["joint_pos"]), "fps": float(data["fps"][0]),
                  "duration_s": (len(data["joint_pos"]) - 1) / float(data["fps"][0]),
                  "head_reference_min_rad": head.min(0).tolist(), "head_reference_max_rad": head.max(0).tolist()}
    audit_path = Path("reports/t800-stage1-motion-audit.json")
    audit = json.loads(audit_path.read_text())
    assert audit["fk"]["passed"] and all(m["passed"] for m in audit["motions"])
    params_path = Path("reports/upstream-audit/t800-source-parameters.json")
    params = json.loads(params_path.read_text())
    assert params["matched"] and params["joint_count"] == NJ
    parity_path = args.parity_report
    parity = json.loads(parity_path.read_text())
    assert parity["passed"] and parity["matmul_precision"] == args.matmul_precision
    assert parity["arguments"]["recompute_mode"] == args.recompute_mode
    assert parity["arguments"]["full_model"] and parity["arguments"]["compile"]
    for name, digest in parity["implementation_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for relative in ("world_model/wm_trainer.py", "imagination/policy_trainer.py", "data/replay_buffer.py"):
        assert sha(ROOT / "src/supermimic" / relative) == sha(ROOT / "vendor/supermimic_upstream/src/supermimic" / relative)
    frozen = [Path("configs/strict-source-lock.json"), Path("configs/sources.lock.json"), motion_path,
              motion_path.with_suffix(".provenance.json"), audit_path, params_path, XML,
              Path("scripts/train_memory.py"), Path("scripts/run_training_stage.py"),
              Path("scripts/evaluate_tracking.py"), parity_path, Path(__file__)]
    frozen += list((ROOT / "optimizations").glob("*.py"))
    memory_report_path = Path(f"reports/t800_storage_bench-{args.matmul_precision}.json")
    memory_report = json.loads(memory_report_path.read_text())
    selected_memory = next(r for r in memory_report["results"] if r["mode"] == args.recompute_mode)
    assert selected_memory["status"] == "passed"
    frozen.append(memory_report_path)
    protocol = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "run_name": args.run_name, "robot": "t800",
        "algorithm_baseline": {"repository": "CuboiLeo/SuperMimic-Reproduced", "commit": lock["baseline_commit"],
                               "recipe": "single-GPU defaults; not the released e8 four-GPU configuration",
                               "config_differences": diff,
                               "byte_identical_training_components": ["wm_trainer", "policy_trainer", "replay_buffer"]},
        "initialization": "random_seed_0", "motion": motion,
        "schema": {"joints": NJ, "qpos": NQ, "qvel": NV, "state": STATE_DIM, "wm_delta": DELTA_DIM,
                   "observation_per_frame": OBS_FRAME_DIM, "goal_per_frame": GOAL_FEAT_DIM,
                   "joint_names": list(JOINT_NAMES), "tracked_bodies": list(TRACKED_BODIES)},
        "adapter_decisions": [
            "Use the existing EngineAI serial T800 model and validated control parameters",
            "Keep the EngineAI 15 tracked bodies including head; body rewards average over 15 instead of G1's 14",
            "Keep all 25 controlled joints; the selected walk reference holds both head joints at zero",
            "Do not import Oli's additional 6D relative-orientation observation or its different goal features",
            "Keep CuboiLeo defaults for noise, domain randomization, rewards, termination and LR schedule",
            "Keep the recorded upstream replay-label behavior; diagnostic correction is not applied",
        ],
        "known_differences": params["remaining_differences"] + [
            "CUDA 12 compatible dependencies differ from upstream CUDA 13",
            "Activation recomputation changes activation storage; validated tolerances, not bitwise training equivalence",
            ("Float32 tensors, no AMP; IEEE matmul/cuDNN override differs from upstream TF32"
             if args.matmul_precision == "ieee" else "Float32 tensors, no AMP; TF32 backend matches upstream"),
            "The original all-path TF32 recomputation failed elementwise tolerance; its diagnostic report is retained",
        ],
        "budget": {"additional_updates": 2000, "original_lr_total_iterations": 320000,
                   "original_lr_decay_iterations": 250000,
                   "on_budget_end": "stop, evaluate and review; no automatic unbounded extension",
                   "stop_on": ["nonfinite update metrics", "training exception or OOM", "evaluation exception", "user interruption"]},
        "evaluation": {"checkpoint_interval_control_steps": 1250, "start_frame": 0,
                       "action_noise": False, "observation_noise": False, "domain_randomization": False,
                       "reset_noise": False, "episodes_per_checkpoint": 1,
                       "wm_horizon": 32, "wm_report_horizons": [1, 8, 32],
                       "fixed_wm_windows": "Collected by the untrained policy in a separate evaluation process",
                       "robustness_seeds_after_nominal_gate": [1, 2, 3, 4, 5],
                       "robustness_reset_noise": True},
        "progression_gate": {"description": "Project criteria for considering more motions, not paper benchmark thresholds",
                             "consecutive_complete_checkpoints": 2, "normalized_return_min": .6,
                             "mean_anchor_aligned_body_position_error_m_max": .15,
                             "joint_position_rmse_rad_max": .35, "hard_joint_limit_fraction_max": .01,
                             "torque_saturation_fraction_max": .05,
                             "robustness_completed_episodes_min": 4, "robustness_episodes": 5,
                             "early_falls": "Expected during early training; not evidence of an implementation defect"},
        "migration_references": [
            {"repository": "zita-ch/supermimic_vibed", "commit": "ee0608f917d08a591356b9bdc7d94bd949157341"},
            {"repository": "liuskywalkerjskd/supermimic_oli", "commit": "2dc90d90e89fb2e097e985c84c8b99a77708fa5f",
             "use": "Reference for dimension derivation, named joints and retargeting; no algorithm code imported"},
        ],
        "execution_optimization": {"recompute_mode": args.recompute_mode, "parity_report": str(parity_path)},
        "memory_selection": {"report": str(memory_report_path), "selected_result": selected_memory},
        "execution_precision": {"matmul": args.matmul_precision, "cudnn": args.matmul_precision, "amp": False,
                                "rejected_tf32_parity": "reports/microbatch/g1-recompute-training-tf32-parity.json",
                                "selected_mode_parity": str(parity_path)},
        "frozen_file_sha256": {str(f.resolve().relative_to(ROOT)): sha(f) for f in frozen},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"protocol": str(args.out), "config_difference_keys": list(diff),
                      "motion": motion, "joint_count": NJ, "tracked_body_count": len(TRACKED_BODIES)}, indent=2))


if __name__ == "__main__":
    main()
