"""Compare pinned upstream and working G1 code in separate Python processes.

Audits source/config differences and deterministic forward/action-gradient parity.
It does not run training or establish convergence equivalence.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/upstream-audit"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(destination):
    import re
    import numpy as np
    import torch
    from dataclasses import asdict
    import supermimic.config
    from supermimic.config import TrainConfig
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.envs.mdp.motion_library import MotionLibrary
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.kinematics.rotations import quat_apply_inverse
    from supermimic.policy.nets import Policy, PolicyCfg, Critic, CriticCfg
    from supermimic.world_model.heads import AnalyticHeads, analytic_observation, flatten_history_term_major
    from supermimic.world_model.world_model import WorldModel, WorldModelCfg
    from supermimic.world_model.state_spec import StateSpecCfg
    from supermimic.world_model.wm_trainer import reference_at
    from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
    from mjlab.entity import Entity
    torch.set_num_threads(4)
    cfg = asdict(TrainConfig())
    config_path = Path(destination).with_suffix(".json")
    config_path.write_text(json.dumps({"module":supermimic.config.__file__,"defaults":cfg},indent=2,default=str)+"\n")
    with torch.serialization.safe_globals([Path, type(Path("."))]):
        ck = torch.load(ROOT / ".cache/downloads/e8_boneseed_ckpt_0100000.pt", map_location="cpu", weights_only=True)
    rc = get_g1_robot_cfg()
    model = Entity(rc).spec.compile()
    plain = load_g1_model()
    names = [plain.joint(i).name for i in range(1, plain.njnt)]
    q0 = torch.tensor([next((v for k,v in rc.init_state.joint_pos.items() if re.fullmatch(k,n)),0.) for n in names])
    joint_range = torch.tensor(plain.jnt_range[1:],dtype=torch.float32)
    middle = joint_range.mean(-1)
    half = (joint_range[:,1]-joint_range[:,0])*.45
    limits = torch.stack([middle-half,middle+half],-1)
    heads=AnalyticHeads(G1Kinematics(plain,device="cpu"),TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,limits,q0)
    wc=dict(ck["config"]["world_model"])
    wc["state_spec"]=StateSpecCfg(**wc["state_spec"])
    wm=WorldModel(WorldModelCfg(**wc),heads,"cpu")
    wm.load_state_dict(ck["wm"])
    pc=PolicyCfg(**ck["config"]["policy"])
    policy=Policy(pc)
    policy.load_state_dict(ck["policy"])
    critic=Critic(CriticCfg(**ck["config"]["critic"]),pcfg=pc)
    critic.load_state_dict(ck["critic"])
    lib=MotionLibrary([ROOT/"data/npz/lafan1/walk1_subject1.npz"],TRACKED_BODIES,q0,device="cpu")
    frames=torch.tensor([0,120,300,900])
    ref=lib.ref(frames)
    state=torch.cat([ref.root_pos_w,ref.root_quat_w,ref.joint_pos,ref.root_lin_vel_w,
                     quat_apply_inverse(ref.root_quat_w,ref.root_ang_vel_w),ref.joint_vel],-1)
    xi=torch.zeros(4,33); xi[:,0]=1.; xi[:,1:30]=torch.linspace(-.003,.003,29)
    action0=torch.zeros(4,29)
    obs=flatten_history_term_major(analytic_observation(state,action0,xi,q0)[None].expand(10,-1,-1))
    goal=lib.goal_window(frames,10)
    action=policy(obs,goal).detach().requires_grad_(True)
    pred=wm.step(state,action,xi)
    reference=reference_at(lib,frames)
    reward=heads.reward(pred,action,action0,reference)
    gradient,=torch.autograd.grad(reward["total"].sum(),action)
    value=critic(wm.spec.canonical(state),goal,wm.spec.normalize_xi(xi),heads.root_error_features(state,reference))
    tensors={"goal":goal,"observation":obs,"policy_action":action,"wm_next_state":pred,
             "termination_logits":wm.termination_logits(pred,reference),"critic_value":value,
             "reward_action_gradient":gradient,**{f"reward/{k}":v for k,v in reward.items()}}
    tensors={k:v.detach().cpu() for k,v in tensors.items()}
    for field in ("body_mass","body_inertia","jnt_range","dof_armature","geom_contype","geom_conaffinity",
                  "geom_friction","actuator_trnid","actuator_gainprm","actuator_biasprm","actuator_forcerange"):
        tensors[f"compiled_model/{field}"]=torch.from_numpy(np.array(getattr(model,field)))
    torch.save(tensors,destination)


def flatten(d,prefix=""):
    out={}
    for k,v in d.items():
        key=f"{prefix}.{k}" if prefix else k
        if isinstance(v,dict): out.update(flatten(v,key))
        else: out[key]=v
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--snapshot",type=Path)
    args=p.parse_args()
    if args.snapshot:
        snapshot(args.snapshot)
        return
    OUT.mkdir(parents=True,exist_ok=True)
    source_roots={"upstream":ROOT/"vendor/supermimic_upstream/src","working":ROOT/"src"}
    for name,path in source_roots.items():
        env=os.environ.copy(); env.update(PYTHONPATH=str(path),SUPERMIMIC_ROBOT="g1",MPLCONFIGDIR=str(ROOT/".cache/matplotlib"))
        with (OUT/f"{name}-snapshot.log").open("w") as log:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),"--snapshot",str(OUT/f"{name}-snapshot.pt")],
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import torch
    upstream=torch.load(OUT/"upstream-snapshot.pt",weights_only=True)
    working=torch.load(OUT/"working-snapshot.pt",weights_only=True)
    errors={k:float((upstream[k].double()-working[k].double()).abs().max()) for k in upstream}
    u=json.loads((OUT/"upstream-snapshot.json").read_text())
    w=json.loads((OUT/"working-snapshot.json").read_text())
    uc,wc=flatten(u["defaults"]),flatten(w["defaults"])
    default_diff={k:{"upstream":uc.get(k),"working":wc.get(k)} for k in uc.keys()|wc.keys() if uc.get(k)!=wc.get(k)}
    files=[]
    for relative in sorted({p.relative_to(r) for r in source_roots.values() for p in r.rglob("*.py")}):
        a,b=(r/relative for r in source_roots.values())
        status="added" if not a.exists() else "removed" if not b.exists() else "identical" if a.read_bytes()==b.read_bytes() else "modified"
        files.append({"path":str(relative),"status":status,"upstream_sha256":sha(a) if a.exists() else None,"working_sha256":sha(b) if b.exists() else None})
    runtime_diffs={}
    for run in ("t800_walk2","t800_walk2_h32","g1_standing_label_fix","t800_standing_label_fix","g1_standing_no_critic","t800_standing_no_critic"):
        actual=flatten(json.loads((ROOT/"logs"/run/"config.json").read_text()))
        runtime_diffs[run]={k:{"upstream_default":uc.get(k),"actual":v} for k,v in actual.items() if uc.get(k)!=v}
    result={"baseline_commit":"ba6105349069bf040f1fdad60873d13138d505a2",
            "imports":{"upstream":u["module"],"working":w["module"]},
            "g1_default_config_differences":default_diff,
            "g1_numerical_parity":{"scope":"CPU float32, shared public checkpoint, four fixed G1 states; includes compiled physical parameters and reward gradient with respect to action. Not training/optimizer or CUDA compile equivalence.",
                                   "max_abs_errors":errors,"passed":max(errors.values())<1e-6},
            "source_counts":{s:sum(f["status"]==s for f in files) for s in ("identical","modified","added","removed")},
            "source_files":files,"actual_run_vs_upstream_default":runtime_diffs}
    (OUT/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    # Record the documented single-clip recipe, preserving the original defaults.
    baseline=u["defaults"]
    baseline["run_name"]="upstream_g1_walk1_reference"
    baseline["data"]["npz_dir"]="data/npz/lafan1"
    baseline["data"]["clips"]=["walk1_subject1"]
    baseline["policy"]["goal_dim"]=680
    (OUT/"g1-reference-config.json").write_text(json.dumps(baseline,indent=2)+"\n")
    print(json.dumps({"source_counts":result["source_counts"],"default_diff":default_diff,
                      "parity_passed":result["g1_numerical_parity"]["passed"],"max_error":max(errors.values())},indent=2))
    if not result["g1_numerical_parity"]["passed"]: raise SystemExit("G1 numerical parity failed")


if __name__ == "__main__":
    main()
