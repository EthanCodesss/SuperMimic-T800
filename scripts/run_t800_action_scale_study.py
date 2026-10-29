"""Bounded short-motion action-scale intervention, with a preregistered extension gate."""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "reports/t800_action_scale_study"
MOTION = "data/npz/t800_single/walk2_steady.npz"
FIXED = "reports/training/t800_walk2_policy_ieee_stage1/baseline.wm_windows.pt"
CONVERTED = "reports/t800_action_scale_study/fixed_torque_kp.wm_windows.pt"
INITIAL = "logs/t800_walk2_policy_ieee_stage1/ckpt_initial.pt"
BASE2000 = "logs/t800_walk2_policy_ieee_stage1/ckpt_latest.pt"
BASE8000 = "logs/t800_short_noanchor_8000/ckpt_latest.pt"
RUN2000 = "t800_short_torque_kp_2000"
RUN8000 = "t800_short_torque_kp_8000"


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command, log):
    print("RUN", " ".join(map(str, command)), flush=True)
    with log.open("w") as stream:
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def prepare():
    import torch
    from scripts.t800_action_scaling import metadata, checkpoint_metadata
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / "design.json").exists():
        raise FileExistsError("Frozen design already exists; use --run to execute it")
    original, new = metadata("rl_lab"), metadata("torque_kp")
    fixed = torch.load(ROOT / FIXED, map_location="cpu", weights_only=True)
    converted = copy.deepcopy(fixed)
    old_scale = torch.tensor(list(original["scale"].values()))
    new_scale = torch.tensor(list(new["scale"].values()))
    converted["actions"] = fixed["actions"] * old_scale / new_scale
    error = float((converted["actions"] * new_scale - fixed["actions"] * old_scale).abs().max())
    assert error < 1e-6
    assert torch.equal(converted["states"], fixed["states"]) and torch.equal(converted["xi"], fixed["xi"])
    converted["action_parameterization"] = new
    converted["reencoded_from"] = FIXED
    converted["physical_target_max_difference_rad"] = error
    torch.save(converted, ROOT / CONVERTED)
    assert checkpoint_metadata({}) == original
    assert checkpoint_metadata({"execution": {"action_parameterization": new}}) == new
    damaged = copy.deepcopy(new)
    damaged["scale"][damaged["joint_names"][0]] *= 2
    try:
        checkpoint_metadata({"execution": {"action_parameterization": damaged}})
    except ValueError:
        pass
    else:
        raise AssertionError("Corrupted scale metadata was accepted")
    gate = dict(joint_rmse_ratio_max=0.90, normalized_full_clip_return_not_lower=True,
                body_orientation_reward_not_lower=True, nominal_clip_complete=True,
                minimum_completed_reset_episodes=4, reset_episode_count=5,
                interpretation="A screening gate for extension, not a statistical significance test or a necessary condition for benefit")
    template = json.loads((ROOT / "configs/t800-single-motion-policy-ieee-baseline.json").read_text())
    frozen_paths = [ROOT / name for name in template["frozen_file_sha256"] if (ROOT / name).exists()]
    frozen_paths += [ROOT / MOTION, ROOT / FIXED, ROOT / CONVERTED, ROOT / INITIAL, ROOT / BASE2000,
                     Path(__file__), *[ROOT / "scripts" / name for name in
                     ("t800_action_scaling.py", "train_action_scale.py", "run_action_scale_stage.py",
                      "evaluate_action_scale.py", "audit_action_scale.py")]]
    frozen = {str(path.relative_to(ROOT)): sha(path) for path in frozen_paths}
    protocols = {}
    for name, budget, total, resume in ((RUN2000, 2000, 2000, None),
                                      (RUN8000, 6000, 8000, f"logs/{RUN2000}/ckpt_latest.pt")):
        p = copy.deepcopy(template)
        p.update(run_name=name, created_utc=datetime.now(timezone.utc).isoformat(), anchor_obs=False,
                 initialization="resume_torque_kp_2000" if resume else "random_seed_0",
                 action_parameterization=new, frozen_file_sha256=frozen)
        p["budget"].update(additional_updates=budget, target_total_updates=total)
        p["schema"]["goal_dim"] = 600
        p["evaluation"].update(checkpoint_interval_control_steps=1000, checkpoint_interval_updates=2000,
                                 fixed_wm_windows=CONVERTED, robustness_seeds_each_milestone=[1,2,3,4,5],
                                 defer_evaluation=True)
        p["execution_precision"]["cudnn"] = "tf32; no convolutions in trained models"
        p["comparison_limits"] = ["One training seed; reset seeds are not independent training seeds",
                                  "Scale changes physical exploration noise and action-rate regularization together",
                                  "At 2000 both groups start fresh; 8000 groups both resume after 2000 and refill replay",
                                  "Fixed-window actions re-encoded for equal physical targets; H32 off-policy WM error is not the same as on-policy accuracy"]
        if resume:
            p["resume_checkpoint"] = {"path": resume}
        else:
            p["reference_initial_checkpoint"] = {"path": INITIAL, "sha256": sha(ROOT / INITIAL)}
        path = ROOT / "configs" / f"{name}.json"
        if path.exists():
            raise FileExistsError(path)
        write(path, p)
        protocols[name] = str(path.relative_to(ROOT))
    write(OUT / "design.json", dict(created_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis="T800 fixed raw-action scaling is poorly conditioned for the unchanged SuperMimic policy optimizer, noise and action-rate cost",
        motion=MOTION, anchor_obs=False, training_seed=0, maximum_new_policy_updates=8000,
        control_2000=BASE2000, control_8000=BASE8000, reference_initial=INITIAL,
        control_action_parameterization=original, treatment_action_parameterization=new,
        extension_gate=gate, protocols=protocols, frozen_file_sha256=frozen,
        fixed_window_reencoding_max_target_error_rad=error,
        checkpoint_metadata_validation=True))
    print("Prepared frozen design, physical-equivalent WM windows and metadata checks", flush=True)


def evaluate_control(checkpoint, total):
    for seed in range(6):
        label = f"control_{total}" + (f"_reset_seed_{seed}" if seed else "")
        command = [sys.executable, "scripts/evaluate_action_scale.py", "--robot", "t800", "--checkpoint", checkpoint,
                   "--motion", MOTION, "--wm-horizon", "32", "--wm-test-windows", FIXED,
                   "--out", str(OUT / f"{label}.json"), "--seed", str(seed)]
        if seed:
            command.append("--reset-noise")
        run(command, OUT / f"{label}.log")


def train_stage(name, additional, resume=None):
    command = [sys.executable, "scripts/run_action_scale_stage.py", "--robot", "t800", "--motion", MOTION,
               "--run-name", name, "--stop-after-updates", str(additional), "--checkpoint-update-interval", "2000",
               "--matmul-precision", "ieee", "--recompute-mode", "policy", "--action-scale-mode", "torque_kp",
               "--protocol", f"configs/{name}.json", "--defer-evaluation", "--wm-test-windows", CONVERTED,
               "--robustness-seeds", "1", "2", "3", "4", "5"]
    command += ["--resume", resume] if resume else ["--reference-initial", INITIAL]
    run(command, OUT / f"{name}-driver.log")


def execute():
    design = json.loads((OUT / "design.json").read_text())
    assert json.loads((OUT / "control-chain.json").read_text())["passed"]
    for name, expected in design["frozen_file_sha256"].items():
        assert sha(ROOT / name) == expected, name
    if (OUT / "status.json").exists():
        raise FileExistsError("Existing study status; inspect before any restart")
    status = dict(status="running", pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
                  active_stage="control_2000_evaluation", completed=[])
    write(OUT / "status.json", status)
    try:
        evaluate_control(BASE2000, 2000)
        status.update(active_stage=RUN2000)
        write(OUT / "status.json", status)
        train_stage(RUN2000, 2000)
        status["completed"].append(RUN2000)
        baseline = json.loads((OUT / "control_2000.json").read_text())
        treatment = json.loads((ROOT / f"reports/training/{RUN2000}/final.json").read_text())
        robust = json.loads((ROOT / f"reports/training/{RUN2000}/robustness.json").read_text())
        assert treatment["checkpoint_updates"] == 2000
        gate = design["extension_gate"]
        checks = dict(joint_rmse=treatment["joint_position_rmse_rad"] <= gate["joint_rmse_ratio_max"] * baseline["joint_position_rmse_rad"],
                      full_clip_return=treatment["normalized_full_clip_return"] >= baseline["normalized_full_clip_return"],
                      body_orientation=treatment["reward_fraction_of_term_max"]["motion_body_ori"] >= baseline["reward_fraction_of_term_max"]["motion_body_ori"],
                      completed_clip=treatment["completed_clip"],
                      reset_completion=sum(r["completed_clip"] for r in robust) >= gate["minimum_completed_reset_episodes"])
        write(OUT / "gate-result.json", dict(extend=all(checks.values()), checks=checks,
                                            control=baseline, treatment=treatment, criteria=gate))
        if all(checks.values()):
            status.update(active_stage=RUN8000)
            write(OUT / "status.json", status)
            train_stage(RUN8000, 6000, f"logs/{RUN2000}/ckpt_latest.pt")
            status["completed"].append(RUN8000)
            status.update(active_stage="control_8000_evaluation")
            write(OUT / "status.json", status)
            evaluate_control(BASE8000, 8000)
        status.update(status="evaluations_complete", active_stage=None, finished_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / "status.json", status)
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write(OUT / "status.json", status)
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--run", action="store_true")
    args = p.parse_args()
    if args.prepare == args.run:
        p.error("Select exactly one of --prepare or --run")
    os.chdir(ROOT)
    os.environ.update(SUPERMIMIC_ROBOT="t800", MPLCONFIGDIR=str(ROOT / ".cache/matplotlib"),
                      OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1")
    prepare() if args.prepare else execute()
