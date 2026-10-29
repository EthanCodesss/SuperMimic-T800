"""Bounded initial-posture search; identical GMR targets and solver, no training."""
import copy
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/engineai_gmr'))
os.environ['SUPERMIMIC_ROBOT']='t800'
import numpy as np
from scripts.prepare_t800_continuous_reference import quality,summarize
from supermimic.robots.t800 import DEFAULT_POS,JOINT_NAMES
from general_motion_retargeting import GeneralMotionRetargeting
from general_motion_retargeting.utils.lafan1 import load_bvh_file


def main():
    os.chdir(ROOT)
    frames,height=load_bvh_file('data/raw/lafan1/walk1_subject1.bvh')
    definitions=[dict(name='default',pitch=0.,roll=-.15,yaw=0.,elbow=-.25)]
    definitions += [dict(name=f'p{p:g}_y{y:g}',pitch=p,roll=-.3,yaw=y,elbow=-.7) for p in (-1.,0.,1.) for y in (-1.,0.,1.)]
    out=dict(scope='Ten deterministic initial right-arm postures; root seeded at the first reference target. Warm up 20 times on frame zero, then track first 600 raw frames continuously. No target/task/solver changes or per-frame restarts.',candidates=[])
    path=ROOT/'reports/task_alignment_study/ik-initialization-search.json'
    for definition in definitions:
        rt=GeneralMotionRetargeting('bvh_lafan1','engineai_t800',actual_human_height=height,verbose=False)
        q=rt.configuration.data.qpos.copy();q[7:]=[DEFAULT_POS[n] for n in JOINT_NAMES]
        for j,k in [(18,'pitch'),(19,'roll'),(20,'yaw'),(21,'elbow')]:q[7+j]=definition[k]
        rt.update_targets(copy.deepcopy(frames[0]));p0,r0=rt.scaled_human_data['Hips'];q[:3]=p0;q[3:7]=r0
        rt.configuration.update(q)
        for _ in range(20):rt.retarget(copy.deepcopy(frames[0]))
        initial=rt.configuration.data.qpos.copy()
        qs=[];rows=[];costs=[]
        for frame in frames[:600]:
            qs.append(rt.retarget(copy.deepcopy(frame)).copy());rows.append(quality(rt))
            costs.append(sum(float(np.sum((task.cost*task.compute_error(rt.configuration))**2)) for task in rt.tasks1+rt.tasks2))
        result=dict(definition=definition,initial_qpos=q.tolist(),converged_frame0_qpos=initial.tolist(),
            prefix_quality=summarize(rows,np.array(qs),rt.model),mean_weighted_task_cost=float(np.mean(costs)))
        out['candidates'].append(result);path.write_text(json.dumps(out,indent=2)+'\n')
        print(definition['name'],result['mean_weighted_task_cost'],result['prefix_quality']['mean_task_position_m'],result['prefix_quality']['hard_limit_near_001rad_fraction']['J19_SHOULDER_ROLL_R'],flush=True)
    best=min(out['candidates'],key=lambda x:x['mean_weighted_task_cost'])
    out['selected']=best['definition']['name']
    out['selection_rule']='Lowest mean combined squared weighted GMR task residual on the fixed first 600 source frames; evaluated before any T800 training.'
    path.write_text(json.dumps(out,indent=2)+'\n')
    (ROOT/'reports/task_alignment_study/ik-selected-seed.json').write_text(json.dumps(dict(initial_qpos=best['initial_qpos'],definition=best['definition'],selection_rule=out['selection_rule']),indent=2)+'\n')


if __name__=='__main__':main()
