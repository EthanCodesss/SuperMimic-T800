"""Compare eager/compiled imagined-step values and input gradients on saved states."""
import argparse
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
    p.add_argument('--trace',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--fk-compat',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
    select_source('g1');verify_sources()
    compat=None
    if args.fk_compat:
        from scripts.fk_compile_compat import install
        compat=install()
    import torch
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.10,0)
    from scripts.validate_microbatch import setup
    from supermimic.imagination.compiled import ImagStep,REWARD_TERM_NAMES
    from supermimic.world_model.wm_trainer import reference_at
    from mjlab.utils.torch import configure_torch_backends
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    wm,policy,critic,lib=setup('g1','cuda',torch.float32,True,False)
    wm.load_state_dict(ck['wm']);wm.requires_grad_(False)
    del policy,critic,ck
    tr=torch.load(args.trace,map_location='cpu',weights_only=True)
    pairs=[(t,e) for e in range(4) for t in range(10,min(tr['failure_steps'][e] or len(tr['states']),len(tr['states']))-1)]
    sel=torch.linspace(0,len(pairs)-1,64).long().tolist()
    pairs=[pairs[i] for i in sel]
    s=torch.stack([tr['states'][t,e] for t,e in pairs]).cuda()
    a=torch.stack([tr['actions'][t+1,e] for t,e in pairs]).cuda()
    prev=torch.stack([tr['actions'][t,e] for t,e in pairs]).cuda()
    fr=torch.stack([tr['frames'][t,e]+1 for t,e in pairs]).cuda()
    xi=torch.zeros(64,33,device='cuda');xi[:,0]=1.
    ref=reference_at(lib,fr)
    def execute(fn):
        ss=s.clone().requires_grad_();aa=a.clone().requires_grad_()
        out=fn(ss,aa,prev,xi,ref)
        grad=torch.autograd.grad(out[1].sum(),(ss,aa),retain_graph=True)
        ori_grad=torch.autograd.grad(out[2][REWARD_TERM_NAMES.index('motion_body_ori')].sum(),(ss,aa))
        return [x.detach().cpu() for x in out], [x.detach().cpu() for x in grad], [x.detach().cpu() for x in ori_grad]
    def compare(a,b):
        d=a.double()-b.double()
        return dict(max_abs=float(d.abs().max()),relative_l2=float(d.norm()/a.double().norm().clamp_min(1e-30)),
            cosine=float(torch.nn.functional.cosine_similarity(a.double().reshape(1,-1),b.double().reshape(1,-1))))
    rows={};outputs={}
    for precision in ['ieee','tf32']:
        configure_torch_backends(allow_tf32=precision=='tf32')
        eager=ImagStep(wm,lib.default_joint_pos,compile_mode=None)
        compiled=ImagStep(wm,lib.default_joint_pos,compile_mode='default')
        eo,eg,eog=execute(eager);co,cg,cog=execute(compiled)
        labels=['next_state','reward','reward_terms','termination_logit','observation','anchor_pose']
        rows[precision]=dict(outputs={k:compare(x,y) for k,x,y in zip(labels,eo,co)},
            reward_gradient={k:compare(x,y) for k,x,y in zip(['state','action'],eg,cg)},
            body_orientation_reward_gradient={k:compare(x,y) for k,x,y in zip(['state','action'],eog,cog)},
            eager_reward_terms=dict(zip(REWARD_TERM_NAMES,eo[2].mean(-1).tolist())),
            compiled_reward_terms=dict(zip(REWARD_TERM_NAMES,co[2].mean(-1).tolist())))
        outputs[precision]=(co,cg)
        print(precision,rows[precision]['outputs']['reward'],rows[precision]['reward_gradient']['action'],flush=True)
        torch._dynamo.reset()
    io,ig=outputs['ieee'];to,tg=outputs['tf32']
    result=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=sha(args.checkpoint),trace=str(args.trace),batch=64,fk_compile_compat=compat,
        comparisons=rows,compiled_ieee_vs_tf32=dict(reward=compare(io[1],to[1]),
            next_state=compare(io[0],to[0]),action_reward_gradient=compare(ig[1],tg[1])),
        scope='One imagined step on 64 physical states. Same current weights and real analytic heads. Does not establish long-horizon or training convergence equivalence.')
    args.out.write_text(json.dumps(result,indent=2)+'\n');verify_sources()


if __name__=='__main__':main()
