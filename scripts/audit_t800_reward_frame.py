"""Offline reward-definition audit on unchanged saved trajectories.

This does not change an environment, checkpoint, or training reward. Pre-action
paired poses are scored; these are not simulator post-step episode returns.
"""
import hashlib
import json
import os
from pathlib import Path

os.environ["SUPERMIMIC_ROBOT"] = "t800"
os.environ.setdefault("MPLCONFIGDIR",str(Path(__file__).resolve().parents[1]/".cache/matplotlib"))
import numpy as np
import torch
from supermimic.robots.t800 import TRACKED_BODIES, JOINT_NAMES, load_t800_model
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.envs.mdp.rewards import TrackingRewardCfg, reanchor_reference, _sq_quat_err

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/t800_action_scale_study/reward-frame-audit.json"


def pose_scores(fk, q, reference, names, anchor):
    cfg=TrackingRewardCfg()
    state=fk.forward(q)
    target=fk.forward(reference)
    ids=fk.body_indices(names)
    a=fk.body_index(anchor)
    pos,quat=reanchor_reference(target["pos"][:,ids],target["quat"][:,ids],target["pos"][:,a],target["quat"][:,a],state["pos"][:,a],state["quat"][:,a])
    return dict(
        global_anchor_position=torch.exp(-((state["pos"][:,a]-target["pos"][:,a])**2).sum(-1)/cfg.std_root_pos**2),
        global_anchor_orientation=torch.exp(-_sq_quat_err(state["quat"][:,a],target["quat"][:,a])/cfg.std_root_ori**2),
        local_body_position=torch.exp(-((state["pos"][:,ids]-pos)**2).sum(-1).mean(-1)/cfg.std_body_pos**2),
        local_body_orientation=torch.exp(-_sq_quat_err(state["quat"][:,ids],quat).mean(-1)/cfg.std_body_ori**2))


def main():
    torch.set_num_threads(2)
    fk=G1Kinematics(load_t800_model(),device="cpu",dtype=torch.float64)
    variants={
        "base_15_current": (TRACKED_BODIES,"LINK_BASE"),
        "base_14_no_head": (tuple(n for n in TRACKED_BODIES if n!="LINK_HEAD_YAW"),"LINK_BASE"),
        "waist_15": (TRACKED_BODIES,"LINK_WAIST_YAW"),
        "waist_14_no_head": (tuple(n for n in TRACKED_BODIES if n!="LINK_HEAD_YAW"),"LINK_WAIST_YAW")}
    paths={"rl_lab_2000":"reports/t800_action_scale_study/control_2000.trajectory.npz",
           "rl_lab_8000":"reports/training/t800_short_noanchor_8000/final.trajectory.npz",
           "torque_kp_2000":"reports/training/t800_short_torque_kp_2000/final.trajectory.npz"}
    final="reports/training/t800_short_torque_kp_8000/final.trajectory.npz"
    if (ROOT/final).exists():
        paths["torque_kp_8000"]=final
    results={}
    for label,path in paths.items():
        trajectory=np.load(ROOT/path)
        ref=torch.as_tensor(trajectory["reference_qpos"],dtype=torch.float64)
        values={}
        for variant,(names,anchor) in variants.items():
            q=torch.tensor(trajectory["qpos"],dtype=torch.float64,requires_grad=True)
            scores=pose_scores(fk,q,ref,names,anchor)
            gradient=torch.autograd.grad(scores["local_body_orientation"].sum(),q)[0][:,7:]
            assert torch.isfinite(gradient).all()
            reference_scores=pose_scores(fk,ref,ref,names,anchor)
            for v in reference_scores.values():
                torch.testing.assert_close(v,torch.ones_like(v),atol=1e-12,rtol=0.)
            values[variant]=dict(anchor=anchor,tracked_bodies=names,
                mean_pose_reward_fraction={k:float(v.mean().detach()) for k,v in scores.items()},
                local_orientation_pose_gradient_mean_abs_per_rad=dict(zip(JOINT_NAMES,gradient.abs().mean(0).tolist())))
        results[label]=dict(trajectory=path,trajectory_sha256=hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),variants=values)
    result=dict(scope="Offline pre-action paired poses, unchanged trajectories; pose reward fractions and state-pose derivatives only. Not post-step episode return, action gradients, physical feasibility or a retraining result.",
                confirmed_differences=dict(upstream_anchor="torso_link",t800_anchor="LINK_BASE",
                    upstream_tracked_body_count=14,t800_tracked_body_count=15,t800_extra_body="LINK_HEAD_YAW",
                    candidate_torso_correspondence="LINK_WAIST_YAW; semantic candidate, not a validated replacement"),
                checks=dict(perfect_reference_scores_one=True,finite_pose_gradients=True),runs=results)
    OUT.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    for label,record in results.items():
        print(label,{k:round(v["mean_pose_reward_fraction"]["local_body_orientation"],5) for k,v in record["variants"].items()})


if __name__=="__main__":
    main()
