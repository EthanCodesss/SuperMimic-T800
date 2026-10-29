"""Render a reference NPZ or recorded physical rollout without a desktop window."""
import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("g1", "t800"), required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=8.)
    args = parser.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    os.environ.setdefault("MUJOCO_GL", "egl")
    import imageio.v2 as imageio
    import mujoco
    import numpy as np
    from supermimic.robots.spec import ROOT_BODY
    if args.robot == "t800":
        from supermimic.robots.t800 import get_spec
    else:
        from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_spec
    spec = get_spec()
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, .1], rgba=[.65, .68, .72, 1])
    spec.worldbody.add_light(pos=[0, -3, 4], dir=[0, 0, -1])
    model = spec.compile()
    model.vis.headlight.ambient[:] = [.5, .5, .5]
    model.vis.headlight.diffuse[:] = [.8, .8, .8]
    model.vis.global_.offwidth = 960
    model.vis.global_.offheight = 720
    data = mujoco.MjData(model)
    with np.load(args.motion) as motion:
        fps = float(motion["fps"][0])
        if "qpos" in motion:
            qpos = motion["qpos"]
        else:
            root = motion["body_names"].tolist().index(ROOT_BODY)
            qpos = np.concatenate([motion["body_pos_w"][:, root], motion["body_quat_w"][:, root], motion["joint_pos"]], axis=1)
    renderer = mujoco.Renderer(model, height=720, width=960)
    camera = mujoco.MjvCamera()
    camera.distance = 3.7 if args.robot == "t800" else 2.6
    camera.azimuth, camera.elevation = 140., -15.
    option = mujoco.MjvOption()
    option.geomgroup[3] = False
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(args.out, fps=fps / 2, codec="libx264", quality=8) as writer:
            for i in range(0, min(len(qpos), int(args.max_seconds * fps)), 2):
                data.qpos[:] = qpos[i]
                mujoco.mj_forward(model, data)
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera, scene_option=option)
                frame = renderer.render()
                writer.append_data(frame)
                if i == 0:
                    imageio.imwrite(args.out.with_suffix(".png"), frame)
    finally:
        renderer.close()
    print(args.out)


if __name__ == "__main__":
    main()
