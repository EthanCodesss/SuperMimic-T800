"""CPU-only reference, actuator, reward coverage and local dynamics diagnostics.

Uses the actual robot entity configuration, without touching training processes.
Local PD rollouts are diagnostic controllers, not policy evaluations. Inverse
dynamics without contacts is meaningful for unloaded arm joints conditional on
the prescribed root trajectory; it does not certify leg/contact feasibility.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=("t800", "g1"), required=True)
    p.add_argument("--motion", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--starts", type=int, default=32)
    args = p.parse_args()
    os.environ["SUPERMIMIC_ROBOT"] = args.robot
    import mujoco
    import numpy as np
    import torch
    from supermimic.data.motion_convert import quat_rotate_inverse
    from supermimic.robots.spec import joint_names, load_robot_model, NQ, NJ, ROOT_BODY
    from scripts.train_strict import verify_sources
    verify_sources()
    torch.set_num_threads(2)
    if args.robot == "t800":
        from scripts.t800_task_adapter import install
        from scripts.t800_action_scaling import install as install_scale
        install()
        scale_spec = install_scale("torque_kp")
    from supermimic.envs.tracking_env_cfg import get_robot_cfg, TRACKED_BODIES, ANCHOR_BODY, ACTION_SCALE, make_env_cfg, SuperMimicEnvCfgArgs
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads, ReferenceFrame
    cfg = get_robot_cfg()
    spec = cfg.build().spec
    # Same plane/robot contact bitmask as the plane terrain environment.
    spec.worldbody.add_geom(name="audit_floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[0., 0., .1], contype=1, conaffinity=1,
                            friction=[1., .005, .0001])
    model = spec.compile()
    env_cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(args.motion)], num_envs=1))
    env_cfg.sim.mujoco.apply(model)
    bare = load_robot_model()
    names = joint_names()
    assert tuple(model.joint(i).name for i in range(1, model.njnt)) == names
    assert model.nq == NQ and model.nv == NJ + 6 and model.nu == NJ
    def resolve(table, name, default=None):
        values = [v for k, v in table.items() if re.fullmatch(k, name)]
        assert len(values) <= 1
        return values[0] if values else default
    scale = np.array([resolve(ACTION_SCALE, n) for n in names])
    q0 = np.array([resolve(cfg.init_state.joint_pos, n, 0.) for n in names])
    aids = np.array([np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(n).id).item() for n in names])
    kp = model.actuator_gainprm[aids, 0]
    kd = -model.actuator_biasprm[aids, 2]
    effort = model.actuator_forcerange[aids, 1]
    assert np.allclose(kp, -model.actuator_biasprm[aids, 1])
    assert not model.actuator_ctrllimited[aids].any()
    soft = model.jnt_range[1:].copy()
    mid = soft.mean(1)
    width = np.diff(soft, axis=1)[:, 0] * cfg.articulation.soft_joint_pos_limit_factor
    soft = np.stack([mid - width / 2, mid + width / 2], -1)
    with np.load(args.motion) as z:
        arr = {k: z[k].copy() for k in z.files}
    body_names = arr["body_names"].tolist()
    root = body_names.index(ROOT_BODY)
    assert tuple(arr["joint_names"].tolist()) == names
    fps = float(arr["fps"][0]); dt = 1 / fps
    assert fps == 50.
    q = np.concatenate([arr["body_pos_w"][:, root], arr["body_quat_w"][:, root], arr["joint_pos"]], -1).astype(float)
    v = np.concatenate([arr["body_lin_vel_w"][:, root],
                        quat_rotate_inverse(q[:, 3:7], arr["body_ang_vel_w"][:, root]), arr["joint_vel"]], -1).astype(float)
    assert np.isfinite(q).all() and np.isfinite(v).all()
    # Robot scene construction must preserve the reference model's kinematics/inertia.
    fields = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia", "jnt_pos", "jnt_axis", "jnt_range")
    model_parity = {f: float(np.max(np.abs(getattr(model, f) - getattr(bare, f)))) for f in fields}
    assert max(model_parity.values()) < 1e-12, model_parity
    bid = np.array([model.body(n).id for n in body_names])
    sample = np.unique(np.linspace(0, len(q) - 1, min(512, len(q))).astype(int))
    data = mujoco.MjData(model)
    err = dict(position_m=0., quaternion_sign_invariant=0., linear_velocity_m_s=0., angular_velocity_rad_s=0.)
    max_ncon = 0; max_nefc = 0
    for i in sample:
        data.qpos[:] = q[i]; data.qvel[:] = v[i]
        mujoco.mj_forward(model, data)
        err["position_m"] = max(err["position_m"], float(abs(data.xpos[bid] - arr["body_pos_w"][i]).max()))
        dq = np.minimum(np.linalg.norm(data.xquat[bid] - arr["body_quat_w"][i], axis=-1), np.linalg.norm(data.xquat[bid] + arr["body_quat_w"][i], axis=-1))
        err["quaternion_sign_invariant"] = max(err["quaternion_sign_invariant"], float(dq.max()))
        for j, b in enumerate(bid):
            vel = np.empty(6)
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, b, vel, 0)
            err["linear_velocity_m_s"] = max(err["linear_velocity_m_s"], float(abs(vel[3:] - arr["body_lin_vel_w"][i, j]).max()))
            err["angular_velocity_rad_s"] = max(err["angular_velocity_rad_s"], float(abs(vel[:3] - arr["body_ang_vel_w"][i, j]).max()))
        max_ncon = max(max_ncon, data.ncon); max_nefc = max(max_nefc, data.nefc)
    assert max(err.values()) < 2e-5, err
    raw = (q[:, 7:] - q0) / scale
    soft_excess = np.maximum(soft[:, 0] - q[:, 7:], 0) + np.maximum(q[:, 7:] - soft[:, 1], 0)
    # First-step reset action-rate transient excluded; all later consecutive frames included.
    action_rate_loss = .02 * np.square(np.diff(raw, axis=0)).sum(1)
    nominal = None
    if args.robot == "t800":
        from supermimic.robots.t800 import ACTUATOR_GROUPS
        nominal = np.zeros(NJ)
        for ids, pk, dk, lim, armature, speed in ACTUATOR_GROUPS:
            for j in ids:
                assert abs(kp[j] - pk) < 1e-9 and abs(kd[j] - dk) < 1e-9
                assert abs(effort[j] - lim) < 1e-9
                assert abs(model.dof_armature[6 + j] - armature) < 1e-9
                nominal[j] = speed
    # Contact-disabled inverse dynamics: arm torques are conditional on the full
    # prescribed trajectory, without an erroneous zero-root-acceleration assumption.
    inverse = spec.compile(); env_cfg.sim.mujoco.apply(inverse)
    inverse.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    acc = np.gradient(v, dt, axis=0)
    invdata = mujoco.MjData(inverse)
    tau = np.zeros((len(q), NJ))
    for i in range(len(q)):
        invdata.qpos[:] = q[i]; invdata.qvel[:] = v[i]; invdata.qacc[:] = acc[i]
        mujoco.mj_inverse(inverse, invdata)
        tau[i] = invdata.qfrc_inverse[6:]
    target_offset = (tau + kd * v[:, 6:]) / kp
    rows = []
    for j, name in enumerate(names):
        rows.append(dict(joint=name, scale=float(scale[j]), kp=float(kp[j]), kd=float(kd[j]),
            effort_nm=float(effort[j]), max_reference_speed_rad_s=float(abs(v[:, 6+j]).max()),
            nominal_speed_rad_s=None if nominal is None else float(nominal[j]),
            speed_exceed_fraction=None if nominal is None else float((abs(v[:, 6+j]) > nominal[j]).mean()),
            max_frame_delta_rad=float(abs(np.diff(q[:, 7+j])).max()),
            reference_std_rad=float(q[:, 7+j].std()), soft_limit_fraction=float((soft_excess[:, j] > 1e-6).mean()),
            reference_action_abs_p99=float(np.quantile(abs(raw[:, j]), .99)),
            reference_action_abs_max=float(abs(raw[:, j]).max()),
            required_no_contact_torque_abs_p99_nm=float(np.quantile(abs(tau[:, j]), .99)),
            required_no_contact_torque_exceed_fraction=float((abs(tau[:, j]) > effort[j]).mean()),
            required_no_contact_pd_offset_abs_p50_rad=float(np.quantile(abs(target_offset[:, j]), .5)),
            required_no_contact_pd_offset_abs_p99_rad=float(np.quantile(abs(target_offset[:, j]), .99))))
    print("reference/actuator audit complete", flush=True)
    # One-joint perturbations identify joints not represented by the body reward.
    fk = G1Kinematics(bare, device="cpu", dtype=torch.float64)
    heads = AnalyticHeads(fk, TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY, torch.tensor(soft), torch.tensor(q0))
    probe_ids = sample[::max(1, len(sample)//32)]
    s = torch.tensor(np.concatenate([q[probe_ids], v[probe_ids]], -1))
    b = heads.robot_bodies(s)
    ref = ReferenceFrame(b["pos"], b["quat"], b["lin_vel"], b["ang_vel"], b["root_pos"], b["root_quat"])
    a = torch.zeros(len(s), NJ, dtype=torch.float64)
    base = heads.reward(s, a, a, ref)
    coverage = []
    with torch.no_grad():
        for j, name in enumerate(names):
            changed = s.clone(); changed[:, 7+j] += .2
            terms = heads.reward(changed, a, a, ref)
            loss = sum(base[k] - terms[k] for k in base if k.startswith("motion_")) / .1
            coverage.append(dict(joint=name, perturbation_rad=.2, mean_normalized_tracking_reward_loss=float(loss.mean()), max_abs_loss=float(loss.abs().max())))
    # Free-root next-reference PD controller: same protocol on G1 and T800.
    # Root-fixed control is only a servo diagnostic and does not imply balance.
    horizon = min(50, len(q)-1)
    starts = np.unique(np.linspace(0, len(q)-horizon-1, args.starts).astype(int))
    cases = {}
    for mode in ("free_root", "reference_root_prescribed"):
        windows = []; sq = np.zeros(NJ); count = 0; sat = 0; total = 0
        contactmax = 0; constraintmax = 0
        for start in starts:
            d = mujoco.MjData(model); d.qpos[:] = q[start]; d.qvel[:] = v[start]
            mujoco.mj_forward(model, d)
            errors = []; rooterrors = []; failed = None; horizons = {}
            for k in range(horizon):
                current = start+k; target = current+1
                d.ctrl[aids] = q[target, 7:]
                for sub in range(4):
                    if mode == "reference_root_prescribed":
                        # Prescribe a smooth root path at 200 Hz, leaving joints dynamic.
                        alpha = sub/4
                        qr = q[current, :7].copy()
                        tangent = np.zeros(6)
                        mujoco.mju_subQuat(tangent[3:], q[target, 3:7], q[current, 3:7])
                        qr[:3] = (1-alpha)*q[current, :3]+alpha*q[target, :3]
                        mujoco.mju_quatIntegrate(qr[3:], tangent[3:], alpha)
                        d.qpos[:7] = qr
                        d.qvel[:6] = (1-alpha)*v[current, :6]+alpha*v[target, :6]
                        d.qacc_warmstart[:] = 0
                    mujoco.mj_step(model, d)
                    sat += int((abs(d.actuator_force[aids]) >= .99*effort).sum()); total += NJ
                    contactmax = max(contactmax, d.ncon); constraintmax = max(constraintmax, d.nefc)
                e = d.qpos[7:]-q[target, 7:]
                sq += e*e; count += 1; errors.append(e)
                rooterrors.append(float(np.linalg.norm(d.qpos[:3]-q[target, :3])))
                if k+1 in (1, 8, 32, 50):
                    horizons[str(k+1)] = dict(joint_rmse_rad=float(np.sqrt(np.mean(e*e))), root_position_error_m=rooterrors[-1])
                rot_actual = np.empty(9); rot_ref = np.empty(9)
                mujoco.mju_quat2Mat(rot_actual, d.qpos[3:7]); mujoco.mju_quat2Mat(rot_ref, q[target, 3:7])
                if mode == "free_root" and (abs(d.qpos[2]-q[target, 2]) > .25 or abs(rot_actual[8]-rot_ref[8]) > .8):
                    failed = k+1; break
            windows.append(dict(start_frame=int(start), start_s=float(start*dt), failure_step=failed,
                completed_steps=len(errors), joint_rmse_rad=float(np.sqrt(np.mean(np.square(errors)))),
                final_root_position_error_m=rooterrors[-1], horizons=horizons))
        cases[mode] = dict(windows=windows, completed_fraction=float(np.mean([w["failure_step"] is None for w in windows])),
            combined_joint_rmse_rad=float(np.sqrt(sq.sum()/count/NJ)), joint_rmse_rad=dict(zip(names, np.sqrt(sq/count).tolist())),
            saturation_fraction=sat/total, max_contacts=contactmax, max_constraints=constraintmax)
        print(mode, cases[mode]["completed_fraction"], cases[mode]["combined_joint_rmse_rad"], flush=True)
    result = dict(robot=args.robot, motion=str(args.motion), motion_sha256=hashlib.sha256(args.motion.read_bytes()).hexdigest(),
        frames=len(q), cpu_only=True, model_reference_parameter_max_errors=model_parity,
        reference_body_state_max_errors=err, reference_body_state_samples=len(sample),
        reference_max_contacts=max_ncon, reference_max_constraints=max_nefc,
        configured_contact_capacity_per_world=env_cfg.sim.nconmax, configured_constraint_capacity_per_world=env_cfg.sim.njmax,
        actuator_indices_in_joint_order=aids.tolist(), action_targets_clamped=False,
        reference_ideal_normalized_reward_no_action_rate=float(1-.4*soft_excess.sum(1).mean()),
        reference_pd_action_rate_normalized_cost_mean=float(action_rate_loss.mean()),
        reference_ideal_normalized_reward_with_reference_pd_action_rate=float((1-.4*soft_excess[1:].sum(1)-action_rate_loss).mean()),
        joints=rows, reward_coverage=coverage, local_pd_rollouts=cases,
        limitations=["CPU MuJoCo local diagnostics, not the GPU policy benchmark or a full-clip feasibility certificate.",
            "Reference-target PD has no learned balance controller, gravity compensation or velocity feedforward; its failure is not proof that a reference is untrackable.",
            "Contact-disabled inverse dynamics leg torques are not valid ground-contact torque requirements. Arm torques assume no external hand contact and the prescribed root trajectory.",
            "Reference-root-prescribed trials receive external support and therefore only isolate joint servos.",
            "Contact/constraint counts cover selected reference states and PD windows, not all possible trained-policy states."])
    verify_sources()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2)+"\n")
    print(args.out, flush=True)


if __name__ == "__main__":
    main()
