"""Check reference motion and differentiable FK against independent MuJoCo results.

This validates reference data, not the ability of any policy to follow it.
"""
import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
import torch

from supermimic.data.motion_convert import quat_rotate_inverse
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.spec import ROBOT, ROOT_BODY, load_robot_model


def geometry_bottom(model, data, geom_id):
    rotation = data.geom_xmat[geom_id].reshape(3, 3)
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    if kind == mujoco.mjtGeom.mjGEOM_BOX:
        return float(data.geom_xpos[geom_id, 2] - np.abs(rotation[2]) @ size)
    if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
        return float(data.geom_xpos[geom_id, 2] - size[0])
    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
        return float(data.geom_xpos[geom_id, 2] - abs(rotation[2, 2]) * size[1] - size[0])
    raise ValueError(f"Unsupported foot geometry type: {kind}")


def fk_check(model, samples=32):
    rng = np.random.default_rng(17)
    q = np.repeat(model.qpos0[None], samples, axis=0)
    q[:, :3] = rng.normal(size=(samples, 3))
    q[:, 3:7] = rng.normal(size=(samples, 4))
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    q[:, 7:] = rng.uniform(model.jnt_range[1:, 0], model.jnt_range[1:, 1], (samples, model.nq - 7))
    v = rng.normal(size=(samples, model.nv))
    fk = G1Kinematics(model, dtype=torch.float64)
    qt = torch.tensor(q, requires_grad=True)
    out = fk(qt, torch.tensor(v))
    errors = dict(position_m=0., quaternion=0., linear_velocity_m_s=0., angular_velocity_rad_s=0.)
    data = mujoco.MjData(model)
    for i in range(samples):
        data.qpos[:] = q[i]
        data.qvel[:] = v[i]
        mujoco.mj_forward(model, data)
        pos = out["pos"][i].detach().numpy()
        quat = out["quat"][i].detach().numpy()
        errors["position_m"] = max(errors["position_m"], float(np.max(abs(pos - data.xpos[1:]))))
        dq = np.minimum(np.linalg.norm(quat - data.xquat[1:], axis=1), np.linalg.norm(quat + data.xquat[1:], axis=1))
        errors["quaternion"] = max(errors["quaternion"], float(dq.max()))
        for b in range(1, model.nbody):
            velocity = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, velocity, 0)
            for key, field, ref in (("linear_velocity_m_s", "lin_vel", velocity[3:]), ("angular_velocity_rad_s", "ang_vel", velocity[:3])):
                errors[key] = max(errors[key], float(np.max(abs(out[field][i, b - 1].detach().numpy() - ref))))
    # Nonzero gradients and a directional finite difference validate use in BPTT.
    weights = torch.tensor(rng.normal(size=out["pos"].shape))
    objective = (out["pos"] * weights).sum()
    grad, = torch.autograd.grad(objective, qt)
    direction = torch.tensor(rng.normal(size=q.shape))
    eps = 1e-6
    with torch.no_grad():
        fd = (((fk(qt + eps * direction)["pos"] - fk(qt - eps * direction)["pos"]) * weights).sum() / (2 * eps)).item()
    ad = (grad * direction).sum().item()
    errors["gradient_direction_abs_error"] = abs(fd - ad)
    errors["samples"] = samples
    errors["passed"] = all(errors[k] < 1e-8 for k in ("position_m", "quaternion", "linear_velocity_m_s", "angular_velocity_rad_s")) and abs(fd - ad) < 1e-5
    return errors


def audit(path, model):
    with np.load(path) as data:
        names = data["body_names"].tolist()
        root = names.index(ROOT_BODY)
        joints = data["joint_pos"]
        if tuple(data["joint_names"].tolist()) != tuple(model.joint(i).name for i in range(1, model.njnt)):
            raise ValueError("Reference joint ordering differs from model")
        qpos = np.concatenate([data["body_pos_w"][:, root], data["body_quat_w"][:, root], joints], axis=1)
        fps = float(data["fps"][0])
    feet = ("LINK_FOOT_L", "LINK_FOOT_R") if ROBOT == "t800" else ("left_ankle_roll_link", "right_ankle_roll_link")
    geoms = [[i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(name).id and model.geom_group[i] == 3] for name in feet]
    # G1 uses named foot collision spheres; do not include its visual mesh.
    if ROBOT == "g1":
        geoms = [[i for i in range(model.ngeom) if model.geom_bodyid[i] == model.body(name).id and "collision" in model.geom(i).name] for name in feet]
    if not all(geoms):
        raise ValueError("No foot collision geometry found")
    mjdata = mujoco.MjData(model)
    heights = []
    for q in qpos:
        mjdata.qpos[:] = q
        mujoco.mj_kinematics(model, mjdata)
        heights.append([min(geometry_bottom(model, mjdata, g) for g in ids) for ids in geoms])
    heights = np.asarray(heights)
    violation = np.maximum(model.jnt_range[1:, 0] - joints, joints - model.jnt_range[1:, 1])
    grounded = heights.min(axis=1)
    report = {
        "source": str(path), "robot": ROBOT, "frames": len(qpos), "fps": fps,
        "duration_s": (len(qpos) - 1) / fps, "finite": bool(np.isfinite(qpos).all()),
        "max_joint_violation_rad": float(np.maximum(violation, 0).max()),
        "joint_violation_frame_fraction": float((violation.max(axis=1) > 1e-4).mean()),
        "foot_min_height_m": heights.min(axis=0).tolist(),
        "lower_foot_height_quantiles_m": np.quantile(grounded, [0, .05, .5, .95, 1]).tolist(),
        "both_feet_above_10cm_fraction": float((grounded > .1).mean()),
        "foot_penetration_beyond_3cm_fraction": float((grounded < -.03).mean()),
        "left_right_contact_fraction_3cm": (abs(heights) < .03).mean(axis=0).tolist(),
        "reference_only": True,
    }
    report["passed"] = report["finite"] and report["max_joint_violation_rad"] < 1e-4 and report["foot_penetration_beyond_3cm_fraction"] == 0 and report["both_feet_above_10cm_fraction"] == 0
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("motions", nargs="*", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    model = load_robot_model()
    result = {"robot": ROBOT, "fk": fk_check(model), "motions": [audit(p, model) for p in args.motions]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    if not result["fk"]["passed"]:
        raise SystemExit("FK validation failed")


if __name__ == "__main__":
    main()
