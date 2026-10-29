"""Frozen-checkpoint diagnostics: physical elbow interventions and goal response.

No optimizer is constructed and no training parameters are updated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ["SUPERMIMIC_ROBOT"]="t800"
os.environ.setdefault("MPLCONFIGDIR",str(ROOT/".cache/matplotlib"))
import numpy as np
import torch
from scripts.train_strict import verify_sources
from scripts.t800_action_scaling import checkpoint_metadata,install,assert_runtime_scale
from supermimic.envs import SuperMimicEnvCfgArgs,make_env
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.world_model.heads import analytic_observation,flatten_history_term_major

MOTION="data/npz/t800_single/walk2_steady.npz"
STARTS=[20,40,60,80,100,120,140,160]


def write(path,obj):
    temp=path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")
    temp.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def intervention(action,goal,scale,alpha,joints,ramp):
    mask=torch.zeros_like(scale)
    mask[list(joints)]=1.
    # The first goal frame starts with q_ref - q_default (25 entries).
    return action+alpha*ramp*mask*(goal[:,:25]/scale-action)


def goal_response(policy,trace,scale,q0,lib):
    indices=torch.tensor(STARTS)
    obs=trace["observation"][indices].cuda()
    goal=trace["goal"][indices].cuda()
    state=trace["state"].cuda()
    prev=trace["prev_action"].cuda()
    xi=trace["xi"].cuda()
    rows=[]
    for t in STARTS:
        hist=analytic_observation(state[t-9:t+1],prev[t-9:t+1],xi[t-9:t+1],q0)
        reconstructed=flatten_history_term_major(hist[:,None,:])[0]
        rows.append(float((reconstructed-trace["observation"][t].cuda()).abs().max()))
    goal_error=float((goal-lib.goal_window(trace["frame"][indices].cuda(),10)).abs().max())
    assert max(rows)<3e-5 and goal_error<3e-5,(rows,goal_error)
    results={}
    for j in (15,16,20,21):
        g=goal.clone().requires_grad_(True)
        y=policy(obs,g)*scale
        grad=torch.autograd.grad(y[:,j].sum(),g)[0].reshape(-1,10,60)[:,:,j].sum(-1)
        direction=torch.zeros_like(goal).reshape(-1,10,60)
        direction[:,:,j]=1.
        direction=direction.reshape_as(goal)
        finite={}
        with torch.no_grad():
            for eps in (.001,.05):
                delta=(policy(obs,goal+eps*direction)-policy(obs,goal-eps*direction))*scale
                fd=delta[:,j]/(2*eps)
                finite[str(eps)]=dict(mean_derivative=float(fd.mean()),per_state_derivative=fd.tolist(),
                    autograd_relative_l2=float((fd-grad).norm()/grad.norm().clamp_min(1e-9)))
        results[str(j)]=dict(autograd_mean_derivative=float(grad.mean()),autograd_per_state=grad.tolist(),finite_differences=finite)
    with torch.no_grad():
        normal=policy(obs,goal)*scale
        ablations={}
        for name,g in (("zero_goal",torch.zeros_like(goal)),("goal_10_frames_later",lib.goal_window((trace["frame"][indices].cuda()+10).clamp_max(198),10))):
            y=policy(obs,g)*scale
            ablations[name]=dict(physical_target_change_rms_rad=float((y-normal).square().mean().sqrt()),
                                 elbow_target_change_rms_rad=float((y[:,[16,21]]-normal[:,[16,21]]).square().mean().sqrt()))
    return dict(scope="Frozen policy, fixed real observations; small position-only reference offsets across all ten goal frames. Not a retraining or an ideal-response requirement.",
                history_max_abs_error=max(rows),goal_max_abs_error=goal_error,joints=results,ablations=ablations)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--smoke",action="store_true")
    args=p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    lock=verify_sources()
    args.out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(args.checkpoint,map_location="cpu",weights_only=False)
    spec=checkpoint_metadata(ck)
    install(spec["mode"])
    ck_hash=sha(args.checkpoint)
    policy,cfg=load_policy_from_checkpoint(args.checkpoint,"cuda")
    policy.requires_grad_(False)
    scale=torch.tensor(list(spec["scale"].values()),device="cuda")
    assert not cfg["env"].get("anchor_obs",False)
    cases=[("baseline",0.,(16,21),10)]+[(f"both_{a:g}",a,(16,21),10) for a in (-.1,.1,.25,.5,1.)]
    cases += [("left_0.25",.25,(16,),10),("right_0.25",.25,(21,),10),("both_0.25_no_ramp",.25,(16,21),1)]
    if args.smoke:
        cases=cases[:1]
    write(args.out/"protocol.json",dict(checkpoint=str(args.checkpoint),checkpoint_sha256=ck_hash,
        checkpoint_updates=ck["num_updates"],source_commit=lock["baseline_commit"],motion=MOTION,
        motion_sha256=sha(ROOT/MOTION),action_parameterization=spec,script_sha256=sha(Path(__file__)),
        interventions=cases,robustness_seeds=[] if args.smoke else [1,2,3,4,5],
        intervention_definition="Only elbow PD targets are blended towards q_ref, ramped over the first ten control steps; other actions come from the unchanged policy responding to actual history.",
        optimizer_steps=0,limitations=["Only two selected 4000-update checkpoints; one training seed","Reset perturbation seeds are explicitly paired within this diagnosis; not identical to previous evaluation RNG histories"]))
    summaries=[]
    initial_states={}
    baseline_trace=None
    for reset_noise in ([False] if args.smoke else [False,True]):
        env=make_env(SuperMimicEnvCfgArgs(npz_files=[MOTION],num_envs=1,episode_length_s=1e5,
            action_noise_std=0.,obs_noise=False,reset_noise=reset_noise,domain_randomization=False,
            sampling_mode="start",record_transitions=True,replay_capacity=1024,seed=0),"cuda:0")
        from mjlab.utils.torch import configure_torch_backends
        configure_torch_backends(allow_tf32=False)
        assert_runtime_scale(env,spec)
        rec=env.recorder_manager.get_term("replay")
        cmd=env.command_manager.get_term("motion")
        q0=env.scene["robot"].data.default_joint_pos[0].clone()
        captured={}
        compute=env.reward_manager.compute
        def capture(dt):
            r=compute(dt)
            captured["state"]=rec.state_io.read()[0].clone()
            captured["terms"]=env.reward_manager.term_rewards[0].clone()
            return r
        env.reward_manager.compute=capture
        try:
            seeds=[0] if not reset_noise else [1,2,3,4,5]
            for seed in seeds:
                selected=cases if not reset_noise else [x for x in cases if x[0] in ("baseline","both_0.25","both_0.5")]
                if not reset_noise:
                    selected=selected+[("baseline_repeat",0.,(16,21),10)]
                for name,alpha,joints,ramp_steps in selected:
                    obs,_=env.reset(seed=seed)
                    initial=rec.state_io.read()[0].cpu()
                    if seed in initial_states:
                        torch.testing.assert_close(initial,initial_states[seed],atol=0.,rtol=0.)
                    else:
                        initial_states[seed]=initial
                    rows={k:[] for k in ("state","next_state","observation","goal","prev_action","nominal_action","action","reference_qpos","xi","frame","reward","reward_terms")}
                    with torch.no_grad():
                        for t in range(201):
                            frame=cmd.frame.clone()
                            ref=cmd.library.ref(frame)
                            raw=policy(obs["policy"],obs["goal"])
                            act=intervention(raw,obs["goal"],scale,alpha,joints,min(1.,(t+1)/ramp_steps))
                            # Assert the intervention means the intended physical PD target.
                            expected=q0+scale*raw
                            expected[:,list(joints)]+=alpha*min(1.,(t+1)/ramp_steps)*(ref.joint_pos[:,list(joints)]-expected[:,list(joints)])
                            torch.testing.assert_close(q0+scale*act,expected,atol=3e-7,rtol=1e-6)
                            fields=dict(state=rec.state_io.read()[0],observation=obs["policy"][0],goal=obs["goal"][0],
                                prev_action=env.action_manager.action[0],nominal_action=raw[0],action=act[0],
                                reference_qpos=torch.cat([ref.root_pos_w,ref.root_quat_w,ref.joint_pos],-1)[0],
                                xi=rec.ctx.read()[0],frame=frame[0])
                            for k,v in fields.items(): rows[k].append(v.detach().clone())
                            obs,r,term,timeout,_=env.step(act)
                            rows["next_state"].append(captured["state"])
                            rows["reward_terms"].append(captured["terms"])
                            rows["reward"].append(r[0].clone())
                            if bool(term[0]|timeout[0]): break
                        else: raise RuntimeError("Missing clip end")
                    trace={k:torch.stack(v).cpu() for k,v in rows.items()}
                    e=trace["state"][:,7:32]-trace["reference_qpos"][:,7:32]
                    names=list(env.reward_manager.active_terms)
                    terms=dict(zip(names,trace["reward_terms"].mean(0).tolist()))
                    result=dict(case=name,seed=seed,reset_noise=reset_noise,alpha=alpha,joints=joints,ramp_steps=ramp_steps,
                        steps=len(e),completed_clip=not bool(term[0]) and bool(env.termination_manager.get_term("clip_end")[0]),
                        full_clip_normalized_return=float(trace["reward"].sum()/(.1*199)),
                        joint_rmse_rad=float(e.square().mean().sqrt()),elbow_rmse_rad=float(e[:,[16,21]].square().mean().sqrt()),
                        left_elbow_rmse_rad=float(e[:,16].square().mean().sqrt()),right_elbow_rmse_rad=float(e[:,21].square().mean().sqrt()),
                        root_position_mean_error_m=float((trace["state"][:,:3]-trace["reference_qpos"][:,:3]).norm(dim=-1).mean()),
                        per_step_reward_terms=terms,body_orientation_fraction=terms["motion_body_ori"]/.02,
                        action_rate_cost=terms["action_rate_l2"])
                    summaries.append(result)
                    if seed==0:
                        torch.save(trace,args.out/f"{name}.trace.pt")
                    if name=="baseline" and seed==0:
                        baseline_trace=trace
                        torch.save(dict(default_pose=q0.cpu(),soft_limits=env.scene["robot"].data.soft_joint_pos_limits[0].cpu()),args.out/"robot-context.pt")
                        write(args.out/"goal-response.json",goal_response(policy,trace,scale,q0,cmd.library))
                    write(args.out/"physical-results.json",summaries)
                    print(json.dumps({k:result[k] for k in ("case","seed","completed_clip","full_clip_normalized_return","joint_rmse_rad","elbow_rmse_rad")}),flush=True)
        finally:
            env.close()
    assert sha(args.checkpoint)==ck_hash
    verify_sources()
    write(args.out/"physical-status.json",dict(status="passed",episodes=len(summaries),checkpoint_unchanged=True,
                                              source_lock_unchanged=True,optimizer_steps=0))


if __name__=="__main__":
    main()
