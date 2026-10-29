"""Plot absolute root paths and heading error, without root-aligning the policy."""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from tracking_replay import reference_frame_indices


def yaw(q):
    w, x, y, z = q.T
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--evaluations", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("MPLCONFIGDIR", str(root / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with np.load(args.motion) as r:
        root_name = "LINK_BASE" if "LINK_BASE" in r["body_names"] else "pelvis"
        i = r["body_names"].tolist().index(root_name)
        rp, rq, fps = r["body_pos_w"][:, i], r["body_quat_w"][:, i], float(r["fps"][0])
        reference_qpos = np.concatenate([rp, rq, r["joint_pos"]], axis=1)
    origin = rp[0, :2].copy()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    axes[0].plot(*(rp[:, :2] - origin).T, "k--", label="reference", linewidth=2)
    axes[0].plot(0, 0, "ko", markersize=5)
    summary = []
    for path in args.evaluations:
        report = json.loads(path.read_text())
        with np.load(path.with_suffix(".trajectory.npz")) as t:
            q = t["qpos"]
            indices, timing = reference_frame_indices(q, reference_qpos, t["root_error_m"])
        n = min(len(q), len(rp))
        q = q[:n]
        diff = yaw(q[:, 3:7]) - yaw(rq[indices[:n]])
        heading = np.degrees(np.arctan2(np.sin(diff), np.cos(diff)))
        label = f"{report.get('checkpoint_updates')} updates"
        line, = axes[0].plot(*(q[:, :2] - origin).T, label=label)
        color = line.get_color()
        axes[0].plot(*(q[-1, :2] - origin), "x", color=color)
        # Break the line at +/-180 degree wrapping; it is not a physical jump.
        display_heading = heading.copy()
        display_heading[1:][np.abs(np.diff(heading)) > 180] = np.nan
        axes[1].plot(np.arange(n) / fps, display_heading, label=label, color=color)
        summary.append({"evaluation": str(path), "updates": report.get("checkpoint_updates"),
                        "mean_abs_heading_error_deg": float(np.abs(heading).mean()),
                        "final_heading_error_deg": float(heading[-1]),
                        "final_recorded_root_error_m": float(np.linalg.norm(q[-1, :3] - rp[indices[n - 1]])),
                        **timing,
                        "recorded_frames": n})
    axes[0].set(title="Root path in world coordinates", xlabel="X from reference start (m)", ylabel="Y from reference start (m)")
    axes[0].axis("equal")
    axes[1].set(title="Root heading error", xlabel="time (s)", ylabel="wrapped error (degrees)", ylim=(-185, 185))
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)
    args.out.with_suffix(".json").write_text(json.dumps({"records": summary,
        "limits": "Recorded pre-action states matched to evaluated reference frames; no XY/yaw alignment; distinct evaluation endpoints for early failures"}, indent=2) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
