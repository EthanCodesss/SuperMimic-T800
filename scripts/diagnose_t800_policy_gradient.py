"""Read-only parameter gradients, reference consistency, and physical continuations."""
import argparse
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SUPERMIMIC_ROBOT']='t800'
from scripts.diagnose_t800_objective import Diagnostic,STARTS,write,sha,serial,verify_sources
import torch
from supermimic.imagination.rollout import imagine
from supermimic.imagination.objective import policy_objective
from supermimic.world_model.wm_trainer import reference_at
from supermimic.kinematics.rotations import quat_error_magnitude


def parameter_probe(d):
    policy=d.policy
    policy.requires_grad_(True)
    names,params=zip(*policy.named_parameters())
    ro=imagine(d.wm,policy,d.lib,d.start,32,obs_noise=False,step_fn=d.step)
    v=d.critic(d.wm.spec.canonical(ro.states[-1]),ro.goals[-1],d.wm.spec.normalize_xi(d.start.xi),ro.e_root[-1])
    reward_objective=policy_objective(ro,None,d.gamma)/32
    total_objective=policy_objective(ro,v,d.gamma)/32
    # Ascent directions; actual training minimizes -J/H with Adam.
    gr=torch.autograd.grad(reward_objective,params,retain_graph=True)
    gt=torch.autograd.grad(total_objective,params)
    del ro
    obs=d.trace['observation'];goal=d.trace['goal']
    pred=d.q0+policy(obs,goal)*d.scale
    ref=d.trace['reference_qpos'][:,7:32]
    elbow_error=(pred[:,[16,21]]-ref[:,[16,21]])
    elbow_objective=-elbow_error.square().mean()
    ge=torch.autograd.grad(elbow_objective,params)
    def norm(gs,mask=None):
        return sum(g.square().sum() for i,g in enumerate(gs) if mask is None or mask[i]).sqrt()
    def dot(a,b): return sum((x*y).sum() for x,y in zip(a,b))
    def cosine(a,b): return float(dot(a,b)/(norm(a)*norm(b)).clamp_min(1e-30))
    results={}
    buffers=dict(policy.named_buffers())
    with torch.no_grad():
        for name,gs in [('reward',gr),('critic',tuple(t-r for t,r in zip(gt,gr))),('total',gt)]:
            gn=norm(gs)
            enc=[n.startswith('encoders.') for n in names]
            dec=[not x for x in enc]
            result=dict(gradient_norm=float(gn),encoder_gradient_norm=float(norm(gs,enc)),decoder_gradient_norm=float(norm(gs,dec)),
                cosine_with_elbow_target_mse_improvement=cosine(gs,ge),
                raw_gradient_elbow_mse_improvement_dot=float(dot(gs,ge)),target_response={})
            # Stateless, symmetric parameter perturbation. No parameter is modified.
            for part,mask in [('all',None),('encoder',enc),('decoder',dec)]:
                eps=.001
                directions=[g/gn.clamp_min(1e-20) if mask is None or mask[i] else torch.zeros_like(g) for i,g in enumerate(gs)]
                ys=[]
                for sign in (-1,1):
                    virtual={n:p+sign*eps*dr for n,p,dr in zip(names,params,directions)}
                    ys.append(torch.func.functional_call(policy,(virtual,buffers),(obs,goal))*d.scale)
                derivative=(ys[1]-ys[0])/(2*eps)
                result['target_response'][part]=dict(mean_elbow_target_derivative=derivative[:,[16,21]].mean(0).tolist(),
                    elbow_target_mse_derivative=float(2*(elbow_error*derivative[:,[16,21]]).mean()),
                    all_joint_target_change_rms_per_unit_gradient=float(derivative.square().mean().sqrt()))
            results[name]=result
    policy.requires_grad_(False)
    return dict(scope='Eight noise-free real start states, full frozen-policy parameter derivative. Symmetric stateless forward probes only; not an optimizer update or the actual randomized training batch.',components=results)


@torch.no_grad()
def reference_check(d):
    frames=torch.arange(199,device='cuda')
    q,v=d.lib.ref_qpos_qvel(frames)
    s=torch.cat([q,v],-1)
    b=d.heads.robot_bodies(s)
    r=d.lib.bodies(frames)
    ori=quat_error_magnitude(b['quat'],r.quat_w)
    rew=d.heads.reward(s,torch.zeros(199,25,device='cuda'),torch.zeros(199,25,device='cuda'),reference_at(d.lib,frames))
    return dict(body_position_rmse_m=float((b['pos']-r.pos_w).square().mean().sqrt()),
        body_orientation_mean_error_rad=float(ori.mean()),body_orientation_max_error_rad=float(ori.max()),
        body_linear_velocity_rmse_m_s=float((b['lin_vel']-r.lin_vel_w).square().mean().sqrt()),
        body_angular_velocity_rmse_rad_s=float((b['ang_vel']-r.ang_vel_w).square().mean().sqrt()),
        reference_pose_analytic_normalized_reward=float(rew['total'].mean()/.1),
        limitation='Kinematic and reward consistency only; does not establish full-body dynamic feasibility.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',required=True,type=Path)
    args=p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    verify_sources()
    d=Diagnostic(args.folder)
    out=dict(checkpoint_sha256=d.protocol['checkpoint_sha256'],script_sha256=sha(Path(__file__)),
        reference=reference_check(d),parameter_probe=parameter_probe(d),continuations=[],
        continuation_definition='Corrections applied for first 32 steps then original policy resumes until clip end; discounted rewards after step 32 compare to critic tail direction, not absolute value calibration.')
    write(args.folder/'parameter-results.json',out)
    print('REFERENCE',out['reference'],flush=True)
    print('PARAMETERS',out['parameter_probe'],flush=True)
    for beta in (-.05,0.,.05,.25,.5):
        r=d.physical_rollout(199-min(STARTS),beta,turn_off_after=32)
        out['continuations'].append(r)
        write(args.folder/'parameter-results.json',out)
        print('CONTINUATION',beta,sum(r['reward_return'])/len(STARTS),flush=True)
    verify_sources()
    assert sha(ROOT/d.protocol['checkpoint'])==d.protocol['checkpoint_sha256']
    out.update(status='passed',optimizer_steps=0,checkpoint_unchanged=True,source_lock_unchanged=True)
    write(args.folder/'parameter-results.json',out)


if __name__=='__main__': main()
