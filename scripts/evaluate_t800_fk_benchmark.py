"""Upstream benchmark procedure with the corrected, checkpoint-verified T800 profile."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--motion',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    from scripts.train_strict import select_source,verify_sources,sha
    verify_sources();select_source('t800')
    from scripts.t800_fk_profile import install,validate_checkpoint
    profile=install()
    import torch
    torch.set_num_threads(4);torch.manual_seed(0);torch.cuda.set_per_process_memory_fraction(.10)
    from supermimic.eval.benchmark import load_policy_from_checkpoint,run_benchmark,summarize
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    validate_checkpoint(ck,profile)
    policy,cfg=load_policy_from_checkpoint(a.checkpoint,'cuda')
    assert not cfg['env']['anchor_obs']
    results=run_benchmark(policy,[str(a.motion)],'cuda:0',num_envs=1,
        future_frames=cfg['env']['future_frames'],anchor_obs=False)
    result=dict(summary=summarize(results),clips=[asdict(x) for x in results],
        checkpoint=str(a.checkpoint),checkpoint_sha256=sha(a.checkpoint),checkpoint_updates=ck['num_updates'],
        profile=profile,motion=str(a.motion),matmul_precision=torch.backends.cuda.matmul.fp32_precision,
        evaluation='Upstream benchmark procedure on T800-adapted source: 14 corresponding bodies, waist anchor, torque/Kp action scale. Different robot/reference from the G1 e1c benchmark.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
    verify_sources();print(json.dumps(result,indent=2))


if __name__=='__main__':main()
