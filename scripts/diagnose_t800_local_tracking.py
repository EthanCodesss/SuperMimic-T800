"""CPU diagnostics; does not change training or claim free-base motion feasibility."""
import json
import os
from pathlib import Path

os.environ.setdefault("SUPERMIMIC_ROBOT", "t800")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-t800-diagnose")
import mujoco
import numpy as np
from supermimic.robots.t800 import ACTION_SCALE, DEFAULT_POS, JOINT_NAMES, get_t800_robot_cfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/t800_anchor_study/local-diagnosis"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    motion = np.load(ROOT / "data/npz/t800_single/walk2_steady.npz")
    ref = motion["joint_pos"].astype(float)
    scale = np.array([ACTION_SCALE[n] for n in JOINT_NAMES])
    default = np.array([DEFAULT_POS[n] for n in JOINT_NAMES])
    raw_target = (ref - default) / scale
    spec = get_t800_robot_cfg().build().spec
    for key in list(spec.keys):
        spec.delete(key)
    spec.delete(spec.joint("floating_base_joint"))
    model = spec.compile()
    assert (model.nq, model.nv, model.nu) == (25, 25, 25)
    model.opt.timestep = .005
    model.geom_contype[:] = 0
    model.geom_conaffinity[:] = 0
    aids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint(n).id)[0]) for n in JOINT_NAMES]
    cases = {}
    for lookahead in (0, 1):
        data = mujoco.MjData(model)
        data.qpos[:] = ref[0]
        data.qvel[:] = motion["joint_vel"][0]
        states, force = [], []
        for t in range(len(ref)):
            states.append(data.qpos.copy())
            data.ctrl[aids] = ref[min(t + lookahead, len(ref) - 1)]
            for _ in range(4):
                mujoco.mj_step(model, data)
                force.append(data.actuator_force[aids].copy())
        states, force = np.asarray(states), np.asarray(force)
        error = states - ref
        cases[f"target_lookahead_{lookahead}"] = {
            "joint_rmse_rad": dict(zip(JOINT_NAMES, np.sqrt((error ** 2).mean(0)).tolist())),
            "combined_rmse_rad": float(np.sqrt((error ** 2).mean())),
            "torque_saturation_fraction_all_substeps": float(np.mean(np.abs(force) >= .99 * model.actuator_forcerange[aids, 1])),
        }
        np.savez_compressed(OUT / f"fixed-base-{lookahead}.npz", qpos=states, reference=ref, torque=force)
    policies = {}
    for name in ("t800_short_noanchor_8000", "t800_short_anchor_8000"):
        tr = np.load(ROOT / f"reports/training/{name}/final.trajectory.npz")
        q, r = tr["qpos"][:, 7:], tr["reference_qpos"][:, 7:]
        error = q - r
        policies[name] = {
            joint: dict(rmse_rad=float(np.sqrt((error[:, i] ** 2).mean())),
                        mean_error_rad=float(error[:, i].mean()),
                        reference_mean_rad=float(r[:, i].mean()), actual_mean_rad=float(q[:, i].mean()),
                        reference_std_rad=float(r[:, i].std()), actual_std_rad=float(q[:, i].std()))
            for i, joint in enumerate(JOINT_NAMES)
        }
    result = {
        "scope": "CPU MuJoCo, actual robot actuator definitions, fixed upright base, gravity on, all contacts disabled; reference position as PD target at 50 Hz, 200 Hz physics. Isolates joint servo tracking; does not establish walking feasibility, GPU parity, or world-model action-gradient accuracy.",
        "cases": cases, "policies": policies,
        "reference_pd_raw_actions": {n: dict(scale=float(scale[i]), default_rad=float(default[i]),
            minimum=float(raw_target[:, i].min()), maximum=float(raw_target[:, i].max()),
            rms=float(np.sqrt((raw_target[:, i] ** 2).mean())),
            delta_rms=float(np.sqrt((np.diff(raw_target[:, i]) ** 2).mean()))) for i, n in enumerate(JOINT_NAMES)},
    }
    (OUT / "diagnosis.json").write_text(json.dumps(result, indent=2) + "\n")
    for key, case in cases.items():
        print(key, case["combined_rmse_rad"], {n:case["joint_rmse_rad"][n] for n in ("J16_ELBOW_PITCH_L", "J21_ELBOW_PITCH_R", "J03_KNEE_PITCH_L")})
    print(OUT / "diagnosis.json")


if __name__ == "__main__":
    main()
