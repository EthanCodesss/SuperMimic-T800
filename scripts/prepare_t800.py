"""Convert the vendor example and retarget short LAFAN1 clips with pinned GMR.

Writes raw qpos CSV (m, xyzw, radians, SDK order), then 50 Hz NPZ with MuJoCo
body poses/velocities. Raw data stays local; provenance accompanies each NPZ.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from supermimic.data.motion_convert import forward_kinematics, resample_motion
from supermimic.robots.t800 import JOINT_NAMES, REPO_ROOT, load_t800_model


def convert(raw, source, destination, input_fps=30.0):
    if raw.ndim != 2 or raw.shape[1] != 32 or not np.isfinite(raw).all():
        raise ValueError("T800 CSV must have 32 finite columns")
    model = load_t800_model()
    motion = resample_motion(raw, input_fps, 50.)
    bodies = forward_kinematics(model, motion)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination, fps=np.array([50.], np.float32),
        joint_pos=motion.joint_pos.astype(np.float32),
        joint_vel=motion.joint_vel.astype(np.float32),
        body_pos_w=bodies.pos_w.astype(np.float32), body_quat_w=bodies.quat_w.astype(np.float32),
        body_lin_vel_w=bodies.lin_vel_w.astype(np.float32), body_ang_vel_w=bodies.ang_vel_w.astype(np.float32),
        joint_names=np.array(JOINT_NAMES), body_names=np.array(bodies.body_names),
        source=np.array(str(source)), input_fps=np.array([input_fps]), robot=np.array("t800"),
    )
    return destination


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--bvh", type=Path)
    parser.add_argument("--start", type=int, default=60)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if (args.csv is None) == (args.bvh is None):
        parser.error("Choose exactly one of --csv and --bvh")
    if args.csv:
        raw = np.loadtxt(args.csv, delimiter=",")
        source = str(args.csv)
        provenance = {"source": source, "input_fps": 30, "joint_order": list(JOINT_NAMES)}
    else:
        sys.path.insert(0, str(REPO_ROOT / "vendor/engineai_gmr"))
        from general_motion_retargeting import GeneralMotionRetargeting
        from general_motion_retargeting.utils.lafan1 import load_bvh_file
        frames, height = load_bvh_file(str(args.bvh))
        selected = frames[args.start:args.start + args.frames]
        if len(selected) != args.frames:
            raise ValueError("Requested frame window exceeds source clip")
        retargeter = GeneralMotionRetargeting(
            "bvh_lafan1", "engineai_t800", actual_human_height=height, verbose=False,
        )
        names = tuple(retargeter.model.joint(i).name for i in range(1, retargeter.model.njnt))
        if names != JOINT_NAMES:
            raise ValueError("GMR joint order differs from the selected model")
        # Let the IK converge on the first pose before emitting the first frame.
        import copy
        for _ in range(20):
            retargeter.retarget(copy.deepcopy(selected[0]))
        qpos = np.array([retargeter.retarget(copy.deepcopy(f)).copy() for f in selected])
        raw = np.concatenate([qpos[:, :3], qpos[:, [4, 5, 6, 3]], qpos[:, 7:]], axis=1)
        source = f"{args.bvh}:frames[{args.start}:{args.start + args.frames}]"
        provenance = {
            "source": source, "gmr_commit": "71e02c0598f6d6c6fe1514bf1d69bc24e82f801e",
            "input_fps": 30, "assumed_human_height_m": height,
            "ik_warmup_iterations": 20, "joint_order": list(JOINT_NAMES),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out.with_suffix(".csv"), raw, delimiter=",")
    convert(raw, source, args.out)
    args.out.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2))
    print(args.out)


if __name__ == "__main__":
    main()
