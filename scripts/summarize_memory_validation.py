"""Summarize the chosen implementation, rejected alternative and measured runs."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def main():
    parity_path = "reports/microbatch/g1-recompute-final-strict-parity.json"
    parity = read(parity_path)
    assert parity["passed"] and not parity["failures"]
    current = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((ROOT / "optimizations").glob("*.py"))}
    assert parity["implementation_sha256"] == current
    runs = {}
    for name in ("g1_memory_verified", "g1_memory_verified_resume", "t800_memory_verified"):
        prefix = f"reports/microbatch/{name}"
        result = read(f"{prefix}/result.json")
        assert result["status"] == "passed"
        assert result["execution"]["implementation_sha256"] == current
        cfg = read(f"{prefix}/config.json")
        baseline_name = "upstream_g1_full_capacity" if result["execution"]["robot"] == "g1" else "t800_full_capacity"
        baseline = read(f"reports/strict/{baseline_name}/config.json")
        differences = {k: {"baseline": baseline.get(k), "actual": v}
                       for k, v in cfg.items() if v != baseline.get(k)}
        assert set(differences) <= {"run_name", "resume"}, differences
        rows = [json.loads(line) for line in (ROOT / prefix / "updates.jsonl").read_text().splitlines()]
        assert len(rows) == result["additional_updates"] == 8
        assert all(math.isfinite(v) for row in rows for key in ("metrics", "world_model_metrics")
                   for v in row[key].values())
        if "resume" in name:
            assert all(result["resume_exact_state_checks"].values())
        runs[name] = {k: v for k, v in result.items() if k not in ("execution", "limits")}
        runs[name]["config_metadata_differences"] = differences
        runs[name]["all_update_metrics_finite"] = True
        runs[name]["report"] = f"{prefix}/result.json"
    errors = parity["comparisons"]
    failed_micro = read("reports/microbatch/g1-cuda-default-shapes.json")
    result = {
        "chosen_implementation": "activation_recompute",
        "entry": "scripts/train_memory.py", "implementation_sha256": current,
        "unchanged": ["Pinned source trees", "PolicyTrainer.update", "4096 environments", "4096 policy batch",
                      "32-step BPTT", "Network widths", "FP32", "WM 10x2048 windows", "Critic 5x4 updates",
                      "Reward/termination", "Noise/domain randomization", "320k LR schedule"],
        "parity": {"report": parity_path, "passed": True, "scope_batch_size": parity["recipe"]["batch_size"],
                   "full_hidden_sizes_and_public_weights": True, "actor_updates": 2, "critic_updates": 40,
                   "original_elementwise_and_relative_update_checks": True,
                   "max_gradient_relative_l2": max(v["relative_l2"] for k, v in errors.items() if "/gradient" in k),
                   "max_parameter_difference_relative_to_update": max(v.get("relative_l2_to_reference_parameter_update", 0) for v in errors.values()),
                   "max_gae_absolute_error": max(v["max_abs"] for k, v in errors.items() if "/gae" in k)},
        "t800_cpu_anchor_parity": "reports/microbatch/t800-anchor-cpu-recompute-fixed.json",
        "rejected_microbatch": {"training_default": False, "report": "reports/microbatch/g1-cuda-default-shapes.json",
                                "max_parameter_difference_relative_to_update": max(v.get("relative_l2_to_reference_parameter_update", 0) for v in failed_micro["comparisons"].values()),
                                "context": "Public pretrained weights, new Adam state; float64 algebra matched, CUDA update tolerance failed"},
        "runs": runs,
        "limits": ["Not bitwise equality or a convergence result", "Direct numerical reference uses batch 1024, capacity runs use batch 4096",
                   "Torch allocator peak excludes Warp/context; device peak is 250ms sampling including desktop",
                   "Resume exactly restores weights/optimizers, but refills replay and does not restore RNG or simulation state",
                   "CUDA 12 compatibility dependencies still differ from upstream CUDA 13 environment"],
    }
    out = ROOT / "reports/memory-optimization.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"chosen": result["chosen_implementation"], "parity": result["parity"],
                      "runs": {k: {f: v[f] for f in ("updates", "torch_peak_allocated_gib", "device_sampled_peak_used_gib")}
                               for k, v in runs.items()}}, indent=2))


if __name__ == "__main__":
    main()
