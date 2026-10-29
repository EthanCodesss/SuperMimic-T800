"""Independent CPU MuJoCo/Dynamo regression for the hinge-enum tracing bug."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_strict import select_source,verify_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--robot',choices=['g1','t800'],required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    select_source(args.robot);verify_sources()
    import mujoco
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from supermimic.kinematics.g1_fk import G1Kinematics
    if args.robot=='g1':
        from supermimic.data.motion_convert import load_g1_model
        model=load_g1_model()
    else:
        from supermimic.robots.t800 import load_t800_model
        model=load_t800_model()
    kind=int(mujoco.mjtJoint.mjJNT_HINGE)
    def enum_branch(x):
        return x*2 if kind==mujoco.mjtJoint.mjJNT_HINGE else x*3
    def int_branch(x):
        return x*2 if kind==int(mujoco.mjtJoint.mjJNT_HINGE) else x*3
    one=torch.ones(1)
    minimal=dict(eager=float(enum_branch(one).item()),compiled=float(torch.compile(enum_branch,backend='eager')(one).item()),
                 corrected_eager=float(int_branch(one).item()),corrected_compiled=float(torch.compile(int_branch,backend='eager')(one).item()))
    rng=np.random.default_rng(42)
    q=np.repeat(model.qpos0[None],16,0)
    q[:,:3]=rng.normal(size=(16,3));q[:,3:7]=rng.normal(size=(16,4));q[:,3:7]/=np.linalg.norm(q[:,3:7],axis=-1,keepdims=True)
    q[:,7:]=rng.uniform(model.jnt_range[1:,0],model.jnt_range[1:,1],(16,model.nq-7))
    v=rng.normal(size=(16,model.nv));qt=torch.tensor(q);vt=torch.tensor(v)
    fk=G1Kinematics(model,dtype=torch.float64)
    oracle={k:[] for k in ['pos','quat','lin_vel','ang_vel']};d=mujoco.MjData(model)
    for qq,vv in zip(q,v):
        d.qpos[:]=qq;d.qvel[:]=vv;mujoco.mj_forward(model,d)
        oracle['pos'].append(d.xpos[1:].copy());oracle['quat'].append(d.xquat[1:].copy())
        vel=np.zeros((model.nbody-1,6))
        for b in range(1,model.nbody):mujoco.mj_objectVelocity(model,d,mujoco.mjtObj.mjOBJ_XBODY,b,vel[b-1],0)
        oracle['lin_vel'].append(vel[:,3:]);oracle['ang_vel'].append(vel[:,:3])
    oracle={k:torch.tensor(np.array(x)) for k,x in oracle.items()}
    weights=torch.tensor(rng.normal(size=oracle['pos'].shape))
    def run(fn):
        qq=qt.clone().requires_grad_();out=fn(qq,vt)
        grad,=torch.autograd.grad((out['pos']*weights).sum(),qq)
        return {k:x.detach() for k,x in out.items()},grad.detach()
    eager,eg=run(fk)
    broken,bg=run(torch.compile(fk,backend='eager'))
    from scripts.fk_compile_compat import install
    meta=install();torch._dynamo.reset()
    fixed,fg=run(torch.compile(fk,backend='eager'))
    def errors(x):
        result={}
        for k,y in x.items():
            if k=='quat':err=torch.minimum((y-oracle[k]).norm(dim=-1),(y+oracle[k]).norm(dim=-1)).max()
            else:err=(y-oracle[k]).abs().max()
            result[k]=float(err)
        return result
    result=dict(robot=args.robot,torch=torch.__version__,backend='eager (Dynamo tracing only; no Inductor required)',
        minimal_reproducer=minimal,samples=16,hinges=model.nq-7,compat=meta,
        eager_vs_mujoco=errors(eager),original_compiled_vs_mujoco=errors(broken),corrected_compiled_vs_mujoco=errors(fixed),
        original_compiled_joint_gradient_norm=float(bg[:,7:].norm()),eager_joint_gradient_norm=float(eg[:,7:].norm()),
        corrected_gradient_max_abs_error=float((fg-eg).abs().max()),
        original_gradient_max_abs_error=float((bg-eg).abs().max()))
    result['passed']=max(result['corrected_compiled_vs_mujoco'].values())<1e-10 and result['corrected_gradient_max_abs_error']<1e-10
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));assert result['passed'];verify_sources()


if __name__=='__main__':main()
