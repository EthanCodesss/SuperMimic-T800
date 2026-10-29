"""Compare the saved baseline, bounded training, and public positive control."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def brief(record):
    result = {k: record[k] for k in ("checkpoint", "checkpoint_iteration", "completed_clip", "steps",
              "reference_frames", "normalized_return", "mean_root_position_error_m", "joint_position_rmse_rad",
              "termination_reasons", "hard_joint_limit_fraction", "torque_saturation_fraction")}
    result["simulated_seconds"] = record["steps"] / 50
    for field in ("world_model_eval_nse_by_horizon", "world_model_fixed_test_nse_by_horizon"):
        values = record[field]
        result[field] = {str(i + 1): values[i] for i in (0, 7, 31) if i < len(values)} if values else None
    return result


def main():
    baseline = read("reports/training/g1_walk1_stage1/baseline.json")
    after = read("reports/g1-bounded-diagnosis-128-eval.json")
    public = read("reports/g1-public-full-walk1.json")
    run = read("reports/microbatch/g1_bounded_diagnosis_128/result.json")
    assert run["status"] == "passed" and run["additional_updates"] == 128
    rows = [json.loads(s) for s in (ROOT / "reports/microbatch/g1_bounded_diagnosis_128/updates.jsonl").read_text().splitlines()]
    finite = all(math.isfinite(v) for r in rows for key in ("metrics", "world_model_metrics") for v in r[key].values())
    assert finite
    result = {
        "long_training": "stopped_by_user; no background long training remains",
        "estimated_original_20300_step_duration_hours": [2, 3],
        "memory": "Full-size G1/T800 updates fit using activation recomputation; see memory-optimization.json",
        "saved_baseline": brief(baseline), "after_128_additional_updates": brief(after),
        "public_checkpoint_positive_control": brief(public),
        "bounded_training": {"additional_updates": 128, "total_updates": run["updates"],
                             "iteration": run["iteration"], "elapsed_seconds": run["elapsed_seconds"],
                             "all_update_metrics_finite": finite,
                             "torch_peak_allocated_gib": run["torch_peak_allocated_gib"],
                             "device_sampled_peak_used_gib": run["device_sampled_peak_used_gib"]},
        "limitations": [
            "Only one deterministic rollout per checkpoint; no multi-seed success-rate claim",
            "Public weights demonstrate sustained G1 locomotion on the same clip/environment, but have global root-position drift",
            "Finite short-run updates and lower model error do not prove learned policy convergence",
            "G1 positive control does not establish T800 adaptation correctness or T800 convergence",
            "Interrupted long run had no newly saved checkpoint; bounded check resumes the last verified saved checkpoint",
        ],
    }
    (ROOT / "reports/bounded-diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    for key in ("saved_baseline", "after_128_additional_updates", "public_checkpoint_positive_control"):
        item = result[key]
        print(key, item["simulated_seconds"], item["normalized_return"], item["completed_clip"])


if __name__ == "__main__":
    main()
