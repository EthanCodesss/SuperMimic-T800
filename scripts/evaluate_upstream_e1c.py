"""Evaluate the G1 checkpoint with the pristine pinned upstream benchmark."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.train_strict import select_source,verify_sources,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    lock=verify_sources();source=select_source('g1')
    import torch
    torch.set_num_threads(4);torch.manual_seed(0)
    from supermimic.eval.benchmark import load_policy_from_checkpoint,run_benchmark,summarize
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda')
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert cfg['data']['clips']==['walk1_subject1'] and not cfg['env']['anchor_obs']
    results=run_benchmark(policy,['data/npz/lafan1/walk1_subject1.npz'],'cuda:0',num_envs=1,anchor_obs=False)
    result=dict(summary=summarize(results),clips=[asdict(x) for x in results],checkpoint=str(a.checkpoint),
        checkpoint_updates=ck['num_updates'],checkpoint_sha256=sha(a.checkpoint),
        baseline_commit=lock['baseline_commit'],source_root=str(source),
        evaluation='Unmodified upstream run_benchmark, including default backend precision.',
        matmul_precision=torch.backends.cuda.matmul.fp32_precision,
        target=dict(updates=8104,norm_return=.676,steps=13066,failed=False))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(result,indent=2)+'\n')
    verify_sources();print(json.dumps(result,indent=2))


if __name__=='__main__': main()
