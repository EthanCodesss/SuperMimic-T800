"""Create a synthetic double-support calibration pose, not a mocap benchmark."""
from pathlib import Path
import argparse
import json
import os
import re

import mujoco
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("g1", "t800"), default="t800")
    args = parser.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    from audit_motion import geometry_bottom
    from supermimic.robots.spec import load_robot_model, joint_names
    from supermimic.envs.tracking_env_cfg import get_robot_cfg
    model = load_robot_model()
    names = joint_names()
    init = get_robot_cfg().init_state
    defaults = {n: next((v for k, v in init.joint_pos.items() if re.fullmatch(k, n)), 0.) for n in names}
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0
    data.qpos[:3] = init.pos
    data.qpos[3:7] = [1., 0., 0., 0.]
    data.qpos[7:] = [defaults[n] for n in names]
    mujoco.mj_forward(model, data)
    if args.robot == "t800":
        feet = [i for i in range(model.ngeom) if model.geom(i).name.startswith("LINK_FOOT_") and model.geom_group[i] == 3]
    else:
        feet = [i for i in range(model.ngeom) if re.fullmatch(r"(left|right)_foot[1-7]_collision", model.geom(i).name)]
    offset = .002 - min(geometry_bottom(model, data, i) for i in feet)
    data.qpos[2] += offset
    mujoco.mj_forward(model, data)
    path = Path(f"data/npz/{args.robot}_standing/standing.npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    repeat = lambda x: np.repeat(x[None], 301, axis=0).astype(np.float32)
    np.savez_compressed(path, fps=np.array([50.], np.float32),
        joint_pos=repeat(data.qpos[7:]), joint_vel=np.zeros((301,len(names)), np.float32),
        body_pos_w=repeat(data.xpos[1:]), body_quat_w=repeat(data.xquat[1:]),
        body_lin_vel_w=np.zeros((301,model.nbody-1,3), np.float32),
        body_ang_vel_w=np.zeros((301,model.nbody-1,3), np.float32),
        joint_names=np.array(names), body_names=np.array([model.body(i).name for i in range(1,model.nbody)]),
        robot=np.array(args.robot), source=np.array("synthetic static default pose; floor clearance 2 mm"))
    path.with_suffix(".provenance.json").write_text(json.dumps({
        "kind": "synthetic diagnostic, not retargeted motion", "root_height_m": float(data.qpos[2]),
        "floor_clearance_m": .002, "pose": defaults, "duration_s": 6., "robot": args.robot,
    }, indent=2))
    print(path, "root height", data.qpos[2])


if __name__ == "__main__":
    main()
