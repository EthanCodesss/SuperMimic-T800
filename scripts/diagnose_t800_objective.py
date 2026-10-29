"""Frozen physical/model directional derivatives; no optimizer or checkpoint edits."""
import argparse
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SUPERMIMIC_ROBOT']='t800'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import mujoco
import numpy as np
import torch
from scripts.diagnose_t800_policy_commands import MOTION,STARTS,write,sha
from scripts.train_strict import verify_sources
from scripts.t800_action_scaling import checkpoint_metadata,install
from mjlab.scene import Scene
from mjlab.utils.torch import configure_torch_backends
from supermimic.envs.tracking_env_cfg import make_env_cfg,SuperMimicEnvCfgArgs,TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY
from supermimic.envs.mdp.motion_library import MotionLibrary
from supermimic.eval.benchmark import load_policy_from_checkpoint
from supermimic.data.replay_buffer import StartBatch
from supermimic.imagination.rollout import imagine
from supermimic.imagination.compiled import ImagStep,REWARD_TERM_NAMES
from supermimic.imagination.objective import policy_objective
from supermimic.kinematics.g1_fk import G1Kinematics
from supermimic.robots.spec import load_robot_model,joint_names
from supermimic.policy.nets import Critic,CriticCfg,PolicyCfg
from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
from supermimic.world_model.state_spec import StateSpecCfg
from supermimic.world_model.world_model import WorldModel,WorldModelCfg
from supermimic.world_model.wm_trainer import reference_at


class CorrectedPolicy:
    def __init__(self,policy,scale,beta,bias=None):
        self.policy,self.scale,self.beta,self.bias=policy,scale,beta,bias
        self.mask=torch.zeros_like(scale)
        self.mask[[16,21]]=1
        self.t=0

    def __call__(self,obs,goal):
        self.t+=1
        a=self.policy(obs,goal)
        if self.bias is not None: a=a+self.bias
        return a+self.beta[:,None]*min(self.t/10,1.)*self.mask*(goal[:,:25]/self.scale-a)


def serial(v):
    if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
    if isinstance(v,dict): return {k:serial(x) for k,x in v.items()}
    return v


class Diagnostic:
    def __init__(self,folder):
        import json
        self.folder=folder
        self.protocol=json.loads((folder/'protocol.json').read_text())
        ckpath=ROOT/self.protocol['checkpoint']
        assert sha(ckpath)==self.protocol['checkpoint_sha256']
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        self.spec=checkpoint_metadata(ck)
        install(self.spec['mode'])
        self.policy,cfg=load_policy_from_checkpoint(ckpath,'cuda')
        self.policy.eval().requires_grad_(False)
        self.gamma=cfg['policy_trainer']['gamma']
        self.scale=torch.tensor(list(self.spec['scale'].values()),device='cuda')
        self.trace=torch.load(folder/'baseline.trace.pt',map_location='cuda',weights_only=True)
        ctx=torch.load(folder/'robot-context.pt',map_location='cuda',weights_only=True)
        self.q0=ctx['default_pose']
        self.lib=MotionLibrary([MOTION],TRACKED_BODIES,self.q0,'cuda')
        fk=G1Kinematics(load_robot_model(),device='cuda')
        self.heads=AnalyticHeads(fk,TRACKED_BODIES,ANCHOR_BODY,ROOT_BODY,ctx['soft_limits'],self.q0)
        wc=dict(cfg['world_model'])
        wc['state_spec']=StateSpecCfg(**wc['state_spec'])
        self.wm=WorldModel(WorldModelCfg(**wc),self.heads,'cuda')
        self.wm.load_state_dict(ck['wm'])
        self.wm.eval().requires_grad_(False)
        self.critic=Critic(CriticCfg(**cfg['critic']),pcfg=PolicyCfg(**cfg['policy'])).cuda()
        self.critic.load_state_dict(ck['critic'])
        self.critic.eval().requires_grad_(False)
        self.step=ImagStep(self.wm,self.q0,compile_mode=None)
        i=torch.tensor(STARTS,device='cuda')
        tr=self.trace
        self.start=StartBatch(tr['state'][i],tr['xi'][i],tr['frame'][i],tr['prev_action'][i],
            torch.stack([tr['state'][i-d] for d in range(9,-1,-1)]),
            torch.stack([tr['prev_action'][i-d] for d in range(9,-1,-1)]))
        assert torch.allclose(self.start.xi[:,0],torch.ones(len(STARTS),device='cuda'))
        assert self.start.xi[:,1:].abs().max()<1e-6
        ecfg=make_env_cfg(SuperMimicEnvCfgArgs(npz_files=[MOTION],num_envs=1,
            domain_randomization=False,obs_noise=False,reset_noise=False,action_noise_std=0))
        scene=Scene(ecfg.scene,'cpu')
        self.model=scene.compile()
        ecfg.sim.mujoco.apply(self.model)
        self.aids=[int(np.flatnonzero(self.model.actuator_trnid[:,0]==self.model.joint('robot/'+n).id)[0]) for n in joint_names()]
        self.default_np=self.q0.cpu().numpy()
        self.scale_np=self.scale.cpu().numpy()
        configure_torch_backends(allow_tf32=False)

    def new_data(self):
        ds=[]
        for s in self.start.state.cpu().numpy():
            d=mujoco.MjData(self.model)
            d.qpos[:]=s[:32]
            d.qvel[:]=s[32:]
            ds.append(d)
        return ds

    def advance(self,ds,a):
        actions=a.detach().cpu().numpy()
        for d,act in zip(ds,actions):
            d.ctrl[self.aids]=self.default_np+self.scale_np*act
            for _ in range(4): mujoco.mj_step(self.model,d)
        return torch.tensor(np.stack([np.r_[d.qpos,d.qvel] for d in ds]),device='cuda',dtype=torch.float32)

    @torch.no_grad()
    def validate(self):
        ds=self.new_data()
        errors={}
        idx=torch.tensor(STARTS,device='cuda')
        for t in range(32):
            s=self.advance(ds,self.trace['action'][idx+t])
            target=self.trace['next_state'][idx+t]
            if t+1 in (1,8,32):
                errors[str(t+1)]=dict(joint_position_rmse_rad=float((s[:,7:32]-target[:,7:32]).square().mean().sqrt()),
                    joint_velocity_rmse_rad_s=float((s[:,38:]-target[:,38:]).square().mean().sqrt()),
                    root_position_mean_error_m=float((s[:,:3]-target[:,:3]).norm(dim=-1).mean()))
        # A 32-step replay is sensitive to contact solver differences; report and gate use.
        assert errors['32']['joint_position_rmse_rad']<.002,errors
        assert errors['32']['root_position_mean_error_m']<.002,errors
        base=imagine(self.wm,self.policy,self.lib,self.start,32,obs_noise=False,step_fn=self.step)
        wrapped=imagine(self.wm,CorrectedPolicy(self.policy,self.scale,torch.zeros(len(STARTS),device='cuda')),
            self.lib,self.start,32,obs_noise=False,step_fn=self.step)
        repeat=imagine(self.wm,self.policy,self.lib,self.start,32,obs_noise=False,step_fn=self.step)
        equivalence={k:float((getattr(base,k)-getattr(wrapped,k)).abs().max()) for k in ('states','actions','rewards','survival','goals')}
        self_repeat={k:float((getattr(base,k)-getattr(repeat,k)).abs().max()) for k in equivalence}
        first_step=max(float((getattr(base,k)[:2]-getattr(wrapped,k)[:2]).abs().max()) for k in equivalence)
        print('REPLAY/IDENTITY',errors,equivalence,self_repeat,first_step,flush=True)
        assert first_step<1e-5,first_step
        assert equivalence['rewards']<1e-5,equivalence
        assert all(v<=max(1e-6,10*self_repeat[k]) for k,v in equivalence.items()),(equivalence,self_repeat)
        return dict(cpu_replay_vs_gpu=errors,unmodified_upstream_imagine_max_abs_error=equivalence,
                    repeated_unmodified_imagine_max_abs_error=self_repeat,first_two_step_max_abs_error=first_step)

    def model_rollout(self,H,beta=0.,grad=False,noise=False,seed=0,detail=True):
        torch.manual_seed(seed)
        b=torch.full((len(STARTS),),float(beta),device='cuda',requires_grad=grad)
        bias=torch.zeros(25,device='cuda',requires_grad=grad)
        pol=CorrectedPolicy(self.policy,self.scale,b,bias)
        termlist=[]
        def step(*args):
            out=self.step(*args)
            termlist.append(out[2])
            return out
        with torch.set_grad_enabled(grad):
            ro=imagine(self.wm,pol,self.lib,self.start,H,obs_noise=noise,step_fn=step)
            v=self.critic(self.wm.spec.canonical(ro.states[-1]),ro.goals[-1],self.wm.spec.normalize_xi(self.start.xi),ro.e_root[-1])
            weight=(self.gamma**torch.arange(H,device='cuda'))[:,None]*ro.survival[:-1]
            terms=(torch.stack(termlist)*weight[:,None,:]).sum(0)
            rew=terms.sum(0)
            tail=self.gamma**H*ro.survival[-1]*v
            J=rew+tail
            torch.testing.assert_close(J.mean(),policy_objective(ro,v,self.gamma),atol=2e-6,rtol=2e-6)
            results=dict(beta=beta,horizon=H,noise=noise,seed=seed,reward_return=rew,critic_tail=tail,objective=J,
                survival=ro.survival[-1],terms=dict(zip(REWARD_TERM_NAMES,terms)),
                joint_rmse_rad=(ro.states[:-1,:,7:32]-self.lib.ref(ro.frames[:-1].reshape(-1)).joint_pos.reshape(H,-1,25)).square().mean((0,2)).sqrt())
            if grad:
                components={'reward_return':rew,'critic_tail':tail,'objective':J}
                if detail: components.update(dict(zip(REWARD_TERM_NAMES,terms)))
                derivatives={}
                for name,x in components.items():
                    gb,ga=torch.autograd.grad(x.sum(),(b,bias),retain_graph=True)
                    derivatives[name]=dict(beta=gb,raw_output_bias_mean=ga/len(STARTS))
                results['derivatives']=derivatives
        return serial(results)

    @torch.no_grad()
    def physical_rollout(self,H,beta,turn_off_after=None):
        ds=self.new_data()
        st=self.start
        hist=list(analytic_observation(st.hist_states,st.hist_actions,st.xi,self.q0).unbind(0))
        pol=CorrectedPolicy(self.policy,self.scale,torch.full((len(STARTS),),float(beta),device='cuda'))
        prev=st.prev_action
        alive=torch.ones(len(STARTS),device='cuda')
        term_sum=torch.zeros(8,len(STARTS),device='cuda')
        step_terms=[]
        reward_prefix=torch.zeros(len(STARTS),device='cuda')
        states=[st.state]
        for t in range(H):
            absolute_frame=st.frame+t
            frame=absolute_frame.clamp_max(198)
            ref=reference_at(self.lib,frame)
            goal=self.lib.goal_window(frame,10)
            if turn_off_after is not None and t==turn_off_after: pol.beta.zero_()
            action=pol(flatten_history_term_major(torch.stack(hist)),goal)
            s=self.advance(ds,action)
            terms=self.heads.reward(s,action,prev,ref)
            tv=torch.stack([terms[k] for k in REWARD_TERM_NAMES])
            # No clip-end bootstrapping in physical returns; all main horizons end before frame 198.
            alive=alive*(absolute_frame<=198)
            term_sum+=self.gamma**t*alive*tv
            if t<32: reward_prefix+=self.gamma**t*alive*tv.sum(0)
            step_terms.append(tv)
            alive=alive*(~self.heads.terminated(s,ref)).float()*(absolute_frame<198)
            hist=hist[1:]+[analytic_observation(s,action,st.xi,self.q0)]
            prev=action
            states.append(s)
        frame=(st.frame+H).clamp_max(198)
        ref=reference_at(self.lib,frame)
        v=self.critic(self.wm.spec.canonical(s),self.lib.goal_window(frame,10),self.wm.spec.normalize_xi(st.xi),self.heads.root_error_features(s,ref))
        tail=self.gamma**H*alive*v
        ss=torch.stack(states)
        frames=(st.frame[None,:]+torch.arange(H,device='cuda')[:,None]).clamp_max(198)
        return serial(dict(beta=beta,horizon=H,turn_off_after=turn_off_after,reward_return=term_sum.sum(0),critic_tail=tail,
            objective=term_sum.sum(0)+tail,terms=dict(zip(REWARD_TERM_NAMES,term_sum)),survival=alive,
            first_32_reward_return=reward_prefix,remainder_reward_return=term_sum.sum(0)-reward_prefix,
            joint_rmse_rad=(ss[:-1,:,7:32]-self.lib.ref(frames.reshape(-1)).joint_pos.reshape(H,-1,25)).square().mean((0,2)).sqrt()))

    @torch.no_grad()
    def goal_encoder(self):
        goal=self.trace['goal']
        enc=self.policy.encoders['motion']
        zpre=enc.mlp(goal*enc.scale)
        z=enc(goal)
        return dict(abs_latent_above_099_fraction=float((z.abs()>.99).float().mean()),
            pre_tanh_abs_median=float(zpre.abs().median()),pre_tanh_abs_max=float(zpre.abs().max()),
            latent_std_across_clip=z.std(0).tolist(),mean_tanh_derivative=float((1-z.square()).mean()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--validate-only',action='store_true')
    args=p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    verify_sources()
    d=Diagnostic(args.folder)
    validation=d.validate()
    write(args.folder/'objective-validation.json',validation)
    print('VALIDATION',validation,flush=True)
    if args.validate_only: return
    out=dict(scope='Eight fixed real on-policy states, eager unmodified upstream imagine and objective; only virtual input perturbations, no optimizer updates.',
        starts=STARTS,checkpoint_sha256=d.protocol['checkpoint_sha256'],script_sha256=sha(Path(__file__)),
        gamma=d.gamma,correction='Both elbows, beta blending toward current reference; ten-step ramp from each selected start.',
        validation=validation,goal_encoder=d.goal_encoder(),model=[],physical=[],noise_probe=[],
        limitations=['Small deterministic on-policy diagnostic batch differs from noisy randomized training replay.',
          'Physical directional returns use analytic reward at fully integrated state, matching WM reward timing; full-clip experiment uses actual environment reward.',
          'Critic assumes continuation of its trained policy; a persistent correction is a counterfactual policy.'])
    path=args.folder/'objective-results.json'
    for H in (1,8,32):
        g=d.model_rollout(H,grad=True)
        out['model'].append(g)
        print('GRAD',H,{k:float(np.mean(v['beta'])) for k,v in g['derivatives'].items()},flush=True)
        for beta in (-.05,0.,.05,.25,.5):
            if beta!=0: out['model'].append(d.model_rollout(H,beta))
            v=d.physical_rollout(H,beta)
            out['physical'].append(v)
            print('PHYSICAL',H,beta,float(np.mean(v['reward_return'])),flush=True)
        write(path,out)
    for seed in (1,2,3):
        out['noise_probe'].append(d.model_rollout(32,grad=True,noise=True,seed=seed,detail=False))
        write(path,out)
    assert sha(ROOT/d.protocol['checkpoint'])==d.protocol['checkpoint_sha256']
    verify_sources()
    out['status']='passed'
    out['optimizer_steps']=0
    out['checkpoint_unchanged']=True
    out['source_lock_unchanged']=True
    write(path,out)


if __name__=='__main__': main()
