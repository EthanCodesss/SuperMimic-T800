"""Plot measured training diagnostics and independent rollout evaluations."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    scalar = {}
    for run in ("t800_walk2", "t800_walk2_h32"):
        events = EventAccumulator(f"logs/{run}/tb", size_guidance={"scalars": 0})
        events.Reload()
        for key in events.Tags()["scalars"]:
            scalar.setdefault(key, []).extend((v.step, v.value) for v in events.Scalars(key))
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True, constrained_layout=True)
    for key, label in (("wm/online_nse_1step", "1-step"), ("wm/online_nse_8step", "8-step")):
        values = np.array(sorted(scalar.get(key, [])))
        if len(values): axes[0].plot(values[:, 0], values[:, 1], label=label, linewidth=1.3)
    axes[0].set(yscale="log", ylabel="Normalized squared error", title="T800: world-model error on recent training replay")
    values = np.array(sorted(scalar.get("train/episode_norm_return", [])))
    if len(values): axes[1].plot(values[:, 0], values[:, 1], label="Training episodes (random start)", alpha=.65)
    evaluations = []
    for path in Path("reports").glob("t800-*.json"):
        record = json.loads(path.read_text())
        if "checkpoint_iteration" in record and "normalized_return" in record:
            evaluations.append(record)
    evaluations.sort(key=lambda r: r["checkpoint_iteration"])
    if evaluations:
        steps = [r["checkpoint_iteration"] for r in evaluations]
        axes[1].plot(steps, [r["normalized_return"] for r in evaluations], "o-", label="Deterministic evaluation from clip start")
        axes[1].plot(steps, [r["completion_fraction"] for r in evaluations], "s--", label="Fraction of clip reached")
    axes[1].set(ylabel="Return / fraction", title="Physical simulation: tracking and clip progress")
    for key, label in (("diagnostics/hard_joint_limit_fraction", "Hard joint limit"),
                       ("diagnostics/soft_joint_limit_fraction", "Soft joint limit"),
                       ("diagnostics/torque_saturation_fraction", "Torque saturation")):
        values = np.array(sorted(scalar.get(key, [])))
        if len(values): axes[2].plot(values[:, 0], values[:, 1] * 100, label=label)
    axes[2].set(xlabel="Vectorized environment steps", ylabel="Samples (%)", title="Control-step diagnostics (cumulative within each run)")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
        ax.axvline(2000, color="grey", linestyle=":", alpha=.6)
    fig.savefig("reports/t800-training-curves.png", dpi=180)
    fig.savefig("reports/t800-training-curves.pdf")
    Path("reports/evaluation-comparison.json").write_text(json.dumps(evaluations, indent=2))


if __name__ == "__main__":
    main()
