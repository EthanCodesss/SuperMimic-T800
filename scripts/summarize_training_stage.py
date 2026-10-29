"""Summarize bounded training evidence and plot comparable checkpoint evaluations."""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text())


def selected_horizons(values):
    return {str(h): values[h - 1] for h in (1, 8, 32) if values and len(values) >= h}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-name", required=True)
    args = p.parse_args()
    folder = ROOT / "reports/training" / args.run_name
    status = read_json(folder / "status.json")
    protocol = read_json(ROOT / status["protocol"]) if status.get("protocol") else None
    evaluations = read_json(folder / "evaluations.json") if (folder / "evaluations.json").exists() else []
    runtime_dir = ROOT / "reports/microbatch" / args.run_name
    rows = []
    if (runtime_dir / "updates.jsonl").exists():
        for line in (runtime_dir / "updates.jsonl").read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                break  # An in-flight final line is not a completed update.
    metrics_finite = all(math.isfinite(v) for row in rows
                         for key in ("metrics", "world_model_metrics") for v in row[key].values())
    records = []
    for r in evaluations:
        record = {k: r.get(k) for k in (
            "checkpoint", "checkpoint_iteration", "checkpoint_updates", "steps", "reference_frames",
            "completed_clip", "completion_fraction", "normalized_return", "mean_root_position_error_m",
            "joint_position_rmse_rad", "mean_anchor_aligned_body_position_error_m", "termination_reasons",
            "hard_joint_limit_fraction", "soft_joint_limit_fraction", "torque_saturation_fraction")}
        record["on_policy_wm_nse"] = selected_horizons(r.get("world_model_eval_nse_by_horizon"))
        comparable = r.get("world_model_fixed_test_nse_by_horizon")
        if r.get("checkpoint_updates") == 0:
            comparable = r.get("world_model_eval_nse_by_horizon")
        record["fixed_baseline_wm_nse"] = selected_horizons(comparable)
        records.append(record)
    # Do not count repeated evaluation of the same checkpoint iteration twice.
    unique = {r["checkpoint_iteration"]: r for r in records}
    ordered = [unique[k] for k in sorted(unique)]
    nominal_gate = False
    gate_detail = []
    if protocol:
        gate = protocol["progression_gate"]
        for r in ordered[-gate["consecutive_complete_checkpoints"]:]:
            checks = {
                "completed_clip": bool(r["completed_clip"]),
                "return": r["normalized_return"] >= gate["normalized_return_min"],
                "body_error": r["mean_anchor_aligned_body_position_error_m"] <= gate["mean_anchor_aligned_body_position_error_m_max"],
                "joint_error": r["joint_position_rmse_rad"] <= gate["joint_position_rmse_rad_max"],
                "hard_limits": r["hard_joint_limit_fraction"] <= gate["hard_joint_limit_fraction_max"],
                "torque_saturation": r["torque_saturation_fraction"] <= gate["torque_saturation_fraction_max"],
            }
            gate_detail.append({"iteration": r["checkpoint_iteration"], "checks": checks})
        nominal_gate = len(gate_detail) == gate["consecutive_complete_checkpoints"] and all(
            all(r["checks"].values()) for r in gate_detail)
    robust = [read_json(p) for p in sorted(folder.glob("robustness_seed_*.json"))]
    robust_complete = sum(bool(r["completed_clip"]) for r in robust)
    robust_gate = bool(protocol and len(robust) == protocol["progression_gate"]["robustness_episodes"]
                       and robust_complete >= protocol["progression_gate"]["robustness_completed_episodes_min"])
    finished = status["status"] == "finished"
    summary = {
        "run_name": args.run_name, "stage_status": status["status"],
        "recorded_updates": len(rows), "last_update": rows[-1]["update"] if rows else None,
        "all_recorded_update_metrics_finite": metrics_finite,
        "median_update_compute_seconds": statistics.median(r["policy_seconds"] + r["world_model_seconds"] for r in rows[10:]) if len(rows) > 10 else None,
        "training_elapsed_seconds": status.get("training_elapsed_seconds"),
        "torch_peak_allocated_gib": max((r["torch_peak_allocated_gib"] for r in rows), default=None),
        "evaluations": ordered, "nominal_progression_gate_passed": nominal_gate,
        "nominal_progression_gate_details": gate_detail,
        "robustness_completed": robust_complete, "robustness_evaluated": len(robust),
        "ready_for_more_motions": bool(finished and metrics_finite and nominal_gate and robust_gate),
        "interpretation": "Budget completed; review learning trend before extending" if finished else "Stage still running or interrupted; no convergence conclusion",
        "limitations": ["One deterministic first-frame rollout per checkpoint; not a success-rate estimate",
                        "Training seed 0 and one short motion; no unseen-motion generalization claim",
                        "Fixed-baseline WM errors use the same untrained-policy transition windows",
                        "On-policy WM errors use different states/actions as the policy changes",
                        "Joint-limit and saturation fractions sample the last physics substep only",
                        "Progression thresholds are project criteria, not paper benchmarks"],
    }
    if len(ordered) > 1:
        first, last = ordered[0], ordered[-1]
        summary["change_from_initial"] = {k: last[k] - first[k] for k in (
            "completion_fraction", "normalized_return", "mean_root_position_error_m",
            "joint_position_rmse_rad", "mean_anchor_aligned_body_position_error_m")}
    if (runtime_dir / "result.json").exists():
        runtime = read_json(runtime_dir / "result.json")
        summary["runtime"] = {k: runtime.get(k) for k in (
            "status", "iteration", "updates", "additional_updates", "elapsed_seconds", "device_sampled_peak_used_gib")}
    (folder / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if ordered:
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = [r["checkpoint_updates"] for r in ordered]
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = [
            ("Clip progress", "fraction", [("completion_fraction", "progress")]),
            ("Deterministic tracking return", "normalized return", [("normalized_return", "return")]),
            ("Position tracking", "meters", [("mean_root_position_error_m", "root (world)"),
                                               ("mean_anchor_aligned_body_position_error_m", "bodies (anchor aligned)")]),
            ("Joint tracking", "RMSE (rad)", [("joint_position_rmse_rad", "joints")]),
        ]
        for ax, (title, ylabel, fields) in zip(axes.flat, panels):
            for key, label in fields:
                ax.plot(x, [r[key] for r in ordered], "o-", label=label)
            ax.set(title=title, ylabel=ylabel)
            if len(fields) > 1:
                ax.legend()
        ax = axes.flat[4]
        for h in (1, 8, 32):
            ax.plot(x, [r["fixed_baseline_wm_nse"].get(str(h), float("nan")) for r in ordered], "o-", label=f"H={h}")
        ax.set(title="WM error on fixed initial-policy windows", ylabel="normalized squared error", yscale="log")
        ax.legend()
        ax = axes.flat[5]
        for key, label in (("hard_joint_limit_fraction", "hard limits"), ("torque_saturation_fraction", "torque saturation")):
            ax.plot(x, [100 * r[key] for r in ordered], "o-", label=label)
        ax.set(title="Control diagnostics", ylabel="sampled joint states (%)")
        ax.legend()
        for ax in axes.flat:
            ax.set_xlabel("world-model / policy updates")
            ax.grid(alpha=.25)
        fig.suptitle(f"{args.run_name} | one motion, seed 0 | {status['status']}")
        fig.savefig(folder / "learning-curves.png", dpi=160)
        plt.close(fig)
    print(json.dumps({k: summary[k] for k in (
        "stage_status", "recorded_updates", "all_recorded_update_metrics_finite",
        "nominal_progression_gate_passed", "ready_for_more_motions")}, indent=2))


if __name__ == "__main__":
    main()
