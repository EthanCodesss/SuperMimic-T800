"""Read-only G1 checkpoint probes in the pristine upstream GPU environment."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_strict import select_source, verify_sources, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--steps', type=int, default=250)
    args = p.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT/'.cache/matplotlib'))
    source = select_source('g1'); verify_sources()
    import numpy as np
    import torch
    torch.set_num_threads(2); torch.manual_seed(13)
    torch.cuda.set_per_process_memory_fraction(.10, 0)
    from supermimic.envs import make_env, SuperMimicEnvCfgArgs
    from supermimic.envs.state_io import StateIO
    from supermimic.eval.benchmark import load_policy_from_checkpoint
    from supermimic.envs.tracking_env_cfg import TRACKED_BODIES, ANCHOR_BODY, ROOT_BODY
    from supermimic.data.motion_convert import load_g1_model
    from supermimic.kinematics.g1_fk import G1Kinematics
    from supermimic.world_model.heads import AnalyticHeads
    from supermimic.world_model.wm_trainer import reference_at
    from supermimic.envs.mdp.rewards import reanchor_reference
    from mjlab.utils.lab_api.math import quat_error_magnitude
    policy, cfg = load_policy_from_checkpoint(args.checkpoint, 'cuda')
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    assert not cfg['env']['anchor_obs']
    env = make_env(SuperMimicEnvCfgArgs(npz_files=['data/npz/lafan1/walk1_subject1.npz'],
        num_envs=4, episode_length_s=1e5, replay_capacity=512,
        action_noise_std=0., obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode='start', anchor_obs=False, seed=13), 'cuda:0')
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends(allow_tf32=False)
    try:
        cmd = env.command_manager.get_term('motion'); robot=env.scene['robot']
        io=StateIO(env)
        heads=AnalyticHeads(G1Kinematics(load_g1_model(),device='cuda'), TRACKED_BODIES,
            ANCHOR_BODY, ROOT_BODY, robot.data.soft_joint_pos_limits[0], robot.data.default_joint_pos[0])
        starts = torch.tensor([0, 1000, 4000, 8000], device='cuda')
        cmd.forced_frame[:] = starts
        obs,_=env.reset()
        active=torch.ones(4,dtype=torch.bool,device='cuda')
        rewards=[]; qerror=[]; ori=[]; rootpos=[]; actions=[]; last=[]; states=[]; frames=[]
        # Capture reward-time quantities before a possible automatic reset.
        compute=env.reward_manager.compute
        def capture(dt):
            r=compute(dt)
            last.append((env.reward_manager.term_rewards.clone(), io.read(), cmd.frame.clone(), cmd.joint_pos.clone(),
                         quat_error_magnitude(cmd.robot_body_quat_w, cmd.body_quat_relative_w),
                         torch.linalg.vector_norm(cmd.robot_anchor_pos_w-cmd.anchor_pos_w,dim=-1)))
            return r
        env.reward_manager.compute=capture
        with torch.no_grad():
            # Force identical reference/robot states to isolate data/FK/reward interfaces.
            lib=cmd.library; fr=starts+1; ref=lib.ref(fr)
            from supermimic.kinematics.rotations import quat_apply_inverse
            sref=torch.cat([ref.root_pos_w,ref.root_quat_w,ref.joint_pos,ref.root_lin_vel_w,
                           quat_apply_inverse(ref.root_quat_w,ref.root_ang_vel_w),ref.joint_vel],-1)
            ideal=heads.reward(sref,torch.zeros(4,29,device='cuda'),torch.zeros(4,29,device='cuda'),reference_at(lib,fr))
            for step in range(args.steps):
                action=policy(obs['policy'],obs['goal'])
                obs,rew,term,tout,_=env.step(action)
                terms,s,f,refq,bodyori,rooterr=last.pop()
                rewards.append(terms.cpu()); qerror.append(((s[:,7:36]-refq)**2).cpu())
                ori.append(bodyori.cpu()); rootpos.append(rooterr.cpu()); actions.append(action.cpu())
                states.append(s.cpu()); frames.append(f.cpu())
                if step==0: failure=[None]*4
                for i in range(4):
                    if bool(active[i] & (term[i] | tout[i])):
                        failure[i]=step+1; active[i]=False
                if not bool(active.any()):break
        rew=torch.stack(rewards); qe=torch.stack(qerror); ori=torch.stack(ori)
        result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),
            updates=ck.get('num_updates'),source_root=str(source),steps_requested=args.steps,
            starts=starts.cpu().tolist(),matmul_precision=torch.backends.cuda.matmul.fp32_precision,
            ideal_reference_reward_mean={k:float(v.mean()) for k,v in ideal.items()},windows=[])
        for i in range(4):
            n=failure[i] or len(rew); termmean=rew[:n,i].mean(0)
            result['windows'].append(dict(start_frame=int(starts[i]),steps=n,failed=failure[i] is not None,
                normalized_return=float(termmean.sum()/.1),joint_rmse_rad=float(qe[:n,i].mean().sqrt()),
                reward_terms=dict(zip(env.reward_manager.active_terms,termmean.tolist())),
                body_orientation_error_rad=dict(zip(TRACKED_BODIES,ori[:n,i].mean(0).tolist()))))
        args.out.parent.mkdir(parents=True,exist_ok=True)
        torch.save(dict(states=torch.stack(states),frames=torch.stack(frames),actions=torch.stack(actions),
                        joint_squared_error=qe,body_orientation_error=ori,rewards=rew,
                        failure_steps=failure,checkpoint=str(args.checkpoint)),args.out.with_suffix('.trace.pt'))
        result['scope']='Four deterministic 5-second starts, not a full-clip benchmark or training return. Auto-reset samples after first failure are excluded from each window metric.'
        args.out.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
    finally:env.close()
    verify_sources()


if __name__=='__main__':main()
