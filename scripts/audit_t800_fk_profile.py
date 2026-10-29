"""Check the combined T800 profile in real simulation and compiled reward/dynamics gradients."""
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['SUPERMIMIC_ROBOT']='t800'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
import json
import hashlib
import argparse
import numpy as np
import torch
from scripts import t800_task_adapter as adapter
from scripts.t800_action_scaling import install as install_scale,assert_runtime_scale
from scripts.train_strict import verify_sources


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--motion',default='data/npz/t800_walk1_selected_seed/walk1_subject1.npz')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    os.chdir(ROOT)
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.manual_seed(113)
    lock=verify_sources()
    from scripts.t800_fk_profile import install
    profile=install()
    spec=profile['task_adapter'];scale=profile['action_parameterization']
    from supermimic.envs import make_env,SuperMimicEnvCfgArgs
    from supermimic.envs.state_io import StateIO
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.robots.spec import load_robot_model
    from supermimic.world_model.heads import AnalyticHeads,analytic_observation,flatten_history_term_major
    from supermimic.world_model.wm_trainer import reference_at
    from mjlab.tasks.tracking import mdp
    from mjlab.envs import mdp as base_mdp
    from mjlab.managers.scene_entity_config import SceneEntityCfg
    from mjlab.utils.torch import configure_torch_backends
    model=load_robot_model()
    body_info=[]
    for g1,t800 in spec['g1_to_t800'].items():
        bid=model.body(t800).id
        ancestor=[];b=bid
        while b>0:
            ancestor.append(model.body(b).name)
            b=int(model.body_parentid[b])
        joints=[model.joint(j).name for j in range(model.njnt) if model.jnt_bodyid[j]==bid]
        body_info.append(dict(g1=g1,t800=t800,parent=model.body(int(model.body_parentid[bid])).name,
            joints=joints,ancestry=ancestor,body_offset_m=model.body_pos[bid].tolist()))
    assert all('LINK_WAIST_YAW' in x['ancestry'] for x in body_info[8:])
    assert model.joint('J16_ELBOW_PITCH_L').bodyid[0]==model.body('LINK_ELBOW_PITCH_L').id
    motion=args.motion
    env=make_env(SuperMimicEnvCfgArgs(npz_files=[motion],num_envs=8,replay_capacity=64,
        obs_noise=False,action_noise_std=0,domain_randomization=False,reset_noise=False,
        sampling_mode='uniform',seed=13),'cuda:0')
    configure_torch_backends(allow_tf32=False)
    try:
        robot=env.scene['robot'];q0=robot.data.default_joint_pos[0]
        fk=G1Kinematics(model,device='cuda')
        heads=AnalyticHeads(fk,adapter.T800_BODIES,spec['anchor_body'],spec['root_body'],robot.data.soft_joint_pos_limits[0],q0)
        adapter.assert_runtime(env,spec,heads);assert_runtime_scale(env,scale)
        cmd=env.command_manager.get_term('motion');io=StateIO(env)
        cfg=SceneEntityCfg('robot',joint_names=('.*',));cfg.resolve(env.scene)
        obs,_=env.reset();worst={};obs_worst=0.;goal_worst=0.
        with torch.no_grad():
            for t in range(25):
                # Bounded perturbations around the immediate reference PD target.
                action=(cmd.joint_pos-q0)/torch.tensor(list(scale['scale'].values()),device='cuda')
                action=action+.05*torch.randn_like(action)
                obs,*_=env.step(action)
                s=io.read();ref=reference_at(cmd.library,cmd.frame)
                ours=heads.reward(s,env.action_manager.action,env.action_manager.prev_action,ref)
                theirs={
                    'motion_global_root_pos':.5*mdp.motion_global_anchor_position_error_exp(env,'motion',.3),
                    'motion_global_root_ori':.5*mdp.motion_global_anchor_orientation_error_exp(env,'motion',.4),
                    'motion_body_pos':mdp.motion_relative_body_position_error_exp(env,'motion',.3),
                    'motion_body_ori':mdp.motion_relative_body_orientation_error_exp(env,'motion',.4),
                    'motion_body_lin_vel':mdp.motion_global_body_linear_velocity_error_exp(env,'motion',1.),
                    'motion_body_ang_vel':mdp.motion_global_body_angular_velocity_error_exp(env,'motion',3.14),
                    'action_rate_l2':-.1*base_mdp.action_rate_l2(env),
                    'joint_limit':-2*base_mdp.joint_pos_limits(env,cfg),
                }
                for k,v in theirs.items():worst[k]=max(worst.get(k,0.),float((ours[k]/.02-v).abs().max()))
                rec=env.recorder_manager.get_term('replay')
                o=analytic_observation(s,env.action_manager.action,rec.ctx.read(),q0)
                p=0
                for size in (25,25,3,3,25):
                    newest=obs['policy'][:,p*10:(p+size)*10].reshape(8,10,size)[:,-1]
                    obs_worst=max(obs_worst,float((newest-o[:,p:p+size]).abs().max()));p+=size
                goal_worst=max(goal_worst,float((obs['goal']-cmd.library.goal_window(cmd.frame,10)).abs().max()))
        from supermimic.imagination.compiled import ImagStep
        from supermimic.world_model.world_model import WorldModel, WorldModelCfg
        wm=WorldModel(WorldModelCfg(),heads,'cuda')
        # A zero-initialized transition has no action sensitivity. Nonzero
        # diagnostic weights make action-gradient comparisons meaningful.
        with torch.no_grad():
            last=[m for m in wm.dynamics.modules() if isinstance(m,torch.nn.Linear)][-1]
            last.weight.normal_(std=.002)
        wm.requires_grad_(False)
        s=io.read().detach();a=env.action_manager.action.detach();prev=env.action_manager.prev_action.detach()
        xi=rec.ctx.read().detach();ref=reference_at(cmd.library,cmd.frame)
        def execute(fn):
            ss=s.clone().requires_grad_();aa=a.clone().requires_grad_()
            values=fn(ss,aa,prev,xi,ref)
            gradients=torch.autograd.grad(values[1].sum(),(ss,aa))
            return [v.detach() for v in values],[v.detach() for v in gradients]
        eager,eg=execute(ImagStep(wm,q0,compile_mode=None))
        compiled,cg=execute(ImagStep(wm,q0,compile_mode='default'))
        def compare(x,y):
            x=x.double();y=y.double();diff=x-y
            return dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/x.norm().clamp_min(1e-30)),
                        cosine=float(torch.nn.functional.cosine_similarity(x.reshape(1,-1),y.reshape(1,-1))))
        values={k:compare(x,y) for k,x,y in zip(['next_state','reward','reward_terms','termination_logit','observation','anchor_pose'],eager,compiled)}
        gradients={k:compare(x,y) for k,x,y in zip(['state','action'],eg,cg)}
        assert values['reward']['max_abs']<2e-6,values
        assert values['anchor_pose']['max_abs']<2e-5,values
        assert all(x['relative_l2']<1e-3 and x['cosine']>.9999 for x in gradients.values()),gradients
        assert float(eg[0][:,7:32].norm())>1e-6, 'Joint geometry sensitivity must be nonzero'
        assert max(worst.values())<2e-5,worst
        assert obs_worst<2e-5 and goal_worst<2e-5,(obs_worst,goal_worst)
        result=dict(status='passed',source_commit=lock['baseline_commit'],task_adapter=spec,action_parameterization=scale,
            fk_compile_compat=profile['fk_compile_compat'],t800_profile_sha256=profile['t800_profile_sha256'],
            compiled_imagstep_outputs=values,compiled_reward_gradients=gradients,
            joint_state_reward_gradient_norm=float(eg[0][:,7:32].norm()),
            gradient_scope='Full-size randomly initialized dynamics with a nonzero diagnostic output layer; 8 real simulator states; not a trained-policy convergence test.',
            motion=motion,motion_sha256=hashlib.sha256(Path(motion).read_bytes()).hexdigest(),
            mapping_evidence=body_info,unscaled_reward_max_abs_error=worst,observation_newest_frame_max_abs_error=obs_worst,
            goal_max_abs_error=goal_worst,steps=25,num_envs=8,
            physical_control_unchanged=['T800 MJCF','Kp/Kd','effort limits','default pose','collisions','200 Hz physics / 50 Hz control'],
            differences_from_old_adapter=['Anchor LINK_BASE -> LINK_WAIST_YAW','Remove head from tracked-body reward average (15 -> 14)','Elbow samples ELBOW_YAW -> ELBOW_PITCH at flexion hinge'],
            limitations=['Anatomical correspondence does not make G1 and T800 geometry, dynamics or retargeting identical.',
            'Reward parity compares both implementations after forward and command refresh, at a common state/reference instant.'])
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:result[k] for k in ['status','unscaled_reward_max_abs_error','observation_newest_frame_max_abs_error','goal_max_abs_error']},indent=2))
    finally: env.close()
    verify_sources()


if __name__=='__main__': main()
