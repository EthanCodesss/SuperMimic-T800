"""Continuous T800 retargeting with an explicitly recorded initial IK posture.

Only the initial joint seed differs from prepare_t800.py; GMR tasks, weights,
solver, warm-up and the complete source motion are retained.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/engineai_gmr'))
os.environ['SUPERMIMIC_ROBOT']='t800'
import numpy as np
from scripts.prepare_t800 import convert
from supermimic.robots.t800 import JOINT_NAMES,DEFAULT_POS
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def quality(retargeter):
    data=retargeter.configuration.data
    result={}
    for human,task in retargeter.human_body_to_task1.items():
        bid=retargeter.model.body(task.frame_name).id
        target=task.transform_target_to_world.translation()
        error=task.compute_error(retargeter.configuration)
        result[human]=[float(np.linalg.norm(data.xpos[bid]-target)),float(np.linalg.norm(error[3:]))]
    return result


def summarize(rows,qpos,model):
    arrays={k:np.array([r[k] for r in rows]) for k in rows[0]}
    q=qpos[:,7:];lo,hi=model.jnt_range[1:].T
    near=((q-lo<.01)|(hi-q<.01)).mean(0)
    return dict(mean_task_error={k:dict(position_m=float(a[:,0].mean()),orientation_rad=float(a[:,1].mean())) for k,a in arrays.items()},
        hard_limit_near_001rad_fraction=dict(zip(JOINT_NAMES,near.tolist())),
        mean_task_position_m=float(np.mean([a[:,0].mean() for a in arrays.values()])),
        mean_task_orientation_rad=float(np.mean([a[:,1].mean() for a in arrays.values()])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--reference-native-csv',required=True,type=Path)
    p.add_argument('--root-seed',choices=['native','reference'],default='native')
    p.add_argument('--report',type=Path,default=ROOT/'reports/task_alignment_study/continuous-ik-seed-comparison.json')
    p.add_argument('--initial-seed',type=Path)
    args=p.parse_args();os.chdir(ROOT)
    if args.out.exists():raise FileExistsError(args.out)
    bvh=ROOT/'data/raw/lafan1/walk1_subject1.bvh'
    frames,height=load_bvh_file(str(bvh))
    rt=GeneralMotionRetargeting('bvh_lafan1','engineai_t800',actual_human_height=height,verbose=False)
    seed=rt.configuration.data.qpos.copy()
    seed[7:]=np.array([DEFAULT_POS[n] for n in JOINT_NAMES])
    if args.root_seed=='reference':
        rt.update_targets(copy.deepcopy(frames[0]))
        root_pos,root_quat=rt.scaled_human_data[rt.human_root_name]
        seed[:3]=root_pos;seed[3:7]=root_quat
    if args.initial_seed:
        seed=np.array(json.loads(args.initial_seed.read_text())['initial_qpos'],dtype=float)
        assert seed.shape==(rt.model.nq,) and np.isfinite(seed).all()
    rt.configuration.update(seed)
    for _ in range(20):rt.retarget(copy.deepcopy(frames[0]))
    qpos=[];rows=[]
    for i,frame in enumerate(frames):
        qpos.append(rt.retarget(copy.deepcopy(frame)).copy());rows.append(quality(rt))
        if i%1000==0:print('retarget frame',i,flush=True)
    qpos=np.array(qpos)
    candidate_quality=summarize(rows,qpos,rt.model)
    raw=np.loadtxt(args.reference_native_csv,delimiter=',')
    assert len(raw)==len(frames)
    native=np.concatenate([raw[:,:3],raw[:,[6,3,4,5]],raw[:,7:]],-1)
    nrows=[]
    for i,frame in enumerate(frames):
        rt.configuration.update(native[i]);rt.update_targets(copy.deepcopy(frame));nrows.append(quality(rt))
    native_quality=summarize(nrows,native,rt.model)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    raw_candidate=np.concatenate([qpos[:,:3],qpos[:,[4,5,6,3]],qpos[:,7:]],-1)
    np.savetxt(args.out.with_suffix('.csv'),raw_candidate,delimiter=',')
    convert(raw_candidate,f'data/raw/lafan1/walk1_subject1.bvh:all_frames;IK_seed=T800_default_pose;root_seed={args.root_seed}',args.out)
    provenance=dict(source='data/raw/lafan1/walk1_subject1.bvh',source_sha256=hashlib.sha256(bvh.read_bytes()).hexdigest(),
        frames=len(frames),input_fps=30,output_fps=50,human_height_m=height,
        gmr_commit='71e02c0598f6d6c6fe1514bf1d69bc24e82f801e',ik_initial_qpos=seed.tolist(),
        ik_warmup_iterations=20,ik_initialization='T800 validated default joint posture',root_seed=args.root_seed,
        initial_seed_file=str(args.initial_seed) if args.initial_seed else None,
        initial_seed_sha256=hashlib.sha256(args.initial_seed.read_bytes()).hexdigest() if args.initial_seed else None,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        quality=dict(candidate=candidate_quality,native=native_quality),
        limitations=['Task residuals measure fit to the existing GMR human targets; they do not establish full dynamic feasibility.','No per-frame IK restart, filtering or smoothing was applied.'])
    if args.initial_seed:provenance['ik_initialization']='Explicit pretraining selection among ten fixed initial right-arm postures; all other joints use the validated T800 default.'
    args.out.with_suffix('.provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    args.report.write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance['quality'],indent=2))


if __name__=='__main__':main()
