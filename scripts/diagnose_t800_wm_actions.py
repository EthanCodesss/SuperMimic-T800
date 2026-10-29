"""Read-only CPU action-sensitivity check on saved deterministic GPU transitions."""
import json
import os
from pathlib import Path
os.environ.setdefault("SUPERMIMIC_ROBOT", "t800")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-t800-diagnose")
import mujoco
import numpy as np
import torch
from mjlab.scene import Scene
from supermimic.envs.tracking_env_cfg import SuperMimicEnvCfgArgs, make_env_cfg
from supermimic.robots.t800 import JOINT_NAMES, DEFAULT_POS, ACTION_SCALE
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel, WorldModelCfg

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/t800_anchor_study/local-diagnosis"


def main():
    torch.set_num_threads(2)
    cfg = make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[str(ROOT / "data/npz/t800_single/walk2_steady.npz")],
        num_envs=1, domain_randomization=False, obs_noise=False, reset_noise=False, action_noise_std=0))
    scene = Scene(cfg.scene, "cpu")
    model = scene.compile()
    cfg.sim.mujoco.apply(model)
    aids = [int(np.flatnonzero(model.actuator_trnid[:, 0] == model.joint("robot/" + n).id)[0]) for n in JOINT_NAMES]
    scale = np.array([ACTION_SCALE[n] for n in JOINT_NAMES])
    default = np.array([DEFAULT_POS[n] for n in JOINT_NAMES])
    selected = [3, 9, 15, 16, 17, 20, 21, 22]

    def physical_step(state, action):
        data = mujoco.MjData(model)
        data.qpos[:] = state[:32]
        data.qvel[:] = state[32:]
        data.ctrl[aids] = default + scale * action
        for _ in range(4):
            mujoco.mj_step(model, data)
        return np.r_[data.qpos, data.qvel]

    results = {}
    for name in ("t800_short_noanchor_8000", "t800_short_anchor_8000"):
        ck = torch.load(ROOT / f"logs/{name}/ckpt_latest.pt", map_location="cpu", weights_only=False)
        wc = dict(ck["config"]["world_model"])
        wc["state_spec"] = StateSpecCfg(**wc["state_spec"])
        wm = WorldModel(WorldModelCfg(**wc), None, "cpu")
        wm.load_state_dict(ck["wm"])
        del ck
        wm.eval()
        w = torch.load(ROOT / f"reports/training/{name}/final.wm_windows.pt", map_location="cpu", weights_only=True)
        assert torch.allclose(w["xi"][:, 0], torch.ones_like(w["xi"][:, 0]))
        assert w["xi"][:, 1:].abs().max() < 1e-6
        s, a, xi = w["states"][0, :32], w["actions"][0, :32].clone().requires_grad_(True), w["xi"][:32]
        y = wm.step(s, a, xi)
        jac = {}
        for j in selected:
            jac[j] = torch.autograd.grad(y[:, 7+j].sum(), a, retain_graph=True)[0][:, j].detach().numpy()
        with torch.no_grad():
            all_pred = wm.step(w["states"][:-1].reshape(-1, 63), w["actions"].reshape(-1, 25),
                               w["xi"][None].expand(w["actions"].shape[0], -1, -1).reshape(-1, 29))
            all_true = w["states"][1:].reshape(-1, 63)
            q_rmse = (all_pred[:, 7:32] - all_true[:, 7:32]).square().mean(0).sqrt()
        states, actions = s.numpy(), a.detach().numpy()
        truth = w["states"][1, :32].numpy()
        phys_nominal = np.array([physical_step(st, ac) for st, ac in zip(states, actions)])
        differences = {}
        for eps in (.01, .1):
            diffs = {}
            for j in selected:
                fd = []
                for st, ac in zip(states, actions):
                    plus, minus = ac.copy(), ac.copy()
                    plus[j] += eps
                    minus[j] -= eps
                    fd.append((physical_step(st, plus)[7+j] - physical_step(st, minus)[7+j]) / (2*eps))
                fd = np.array(fd)
                g = jac[j]
                diffs[JOINT_NAMES[j]] = dict(physical_mean=float(fd.mean()), wm_mean=float(g.mean()),
                    physical_min=float(fd.min()), physical_max=float(fd.max()),
                    wm_min=float(g.min()), wm_max=float(g.max()),
                    sign_agreement=float(np.mean(np.sign(fd) == np.sign(g))),
                    relative_l2_error=float(np.linalg.norm(g - fd) / max(np.linalg.norm(fd), 1e-12)))
            differences[str(eps)] = diffs
        results[name] = dict(
            cpu_replay_vs_recorded_gpu_joint_rmse_rad=np.sqrt(((phys_nominal[:, 7:32] - truth[:, 7:32])**2).mean(0)).tolist(),
            cpu_replay_vs_recorded_gpu_joint_velocity_rmse_rad_s=np.sqrt(((phys_nominal[:, 38:] - truth[:, 38:])**2).mean(0)).tolist(),
            wm_one_step_joint_rmse_rad=dict(zip(JOINT_NAMES, q_rmse.tolist())),
            diagonal_action_sensitivity=differences, samples=32,
            action_rms=dict(zip(JOINT_NAMES, w["actions"].square().mean((0,1)).sqrt().tolist())),
        )
        print(name, json.dumps(differences["0.1"]), flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result=dict(scope="CPU finite differences versus learned WM autograd, 32 saved on-policy deterministic states per checkpoint, one 20 ms step, diagonal joint-position sensitivity only. CPU/GPU solver and missing warm-start differences are measured; this is diagnostic, not GPU gradient equivalence or a causal ablation.",
                joint_order=list(JOINT_NAMES), runs=results)
    (OUT / "wm-action-sensitivity.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
