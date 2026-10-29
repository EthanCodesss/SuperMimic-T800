"""Make an exactly shared-frame crop from the fully retargeted T800 walk1."""
import hashlib
import json
import argparse
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/npz/t800_walk1_full/walk1_subject1.npz')
    p.add_argument('--destination',type=Path,default=ROOT/'data/npz/t800_walk1_crop/walk1_10s_4s.npz')
    args=p.parse_args()
    source=args.source.resolve();destination=args.destination.resolve()
    if destination.exists(): raise FileExistsError(destination)
    with np.load(source) as data: full={k:data[k] for k in data.files}
    # Chosen before training: same 10 s offset and 3.96 s span as the old short-task protocol.
    start,end=500,699
    fields=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
    crop=dict(full)
    for key in fields: crop[key]=full[key][start:end].copy()
    crop['source']=np.array(str(source.relative_to(ROOT))+f':output_frames[{start}:{end}]')
    destination.parent.mkdir(parents=True)
    np.savez_compressed(destination,**crop)
    with np.load(destination) as data:
        checks={key:bool(np.array_equal(data[key],full[key][start:end])) for key in fields}
    assert all(checks.values())
    result=dict(source=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        crop=str(destination.relative_to(ROOT)),crop_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        full_frames=len(full['joint_pos']),crop_frames=end-start,output_frame_range=[start,end],fps=50,
        crop_rule='First full-clip continuous IK and velocity computation; then slice all six reference arrays without recalculating derivatives or rebasing world coordinates.',
        exact_array_checks=checks,source_provenance=json.loads(source.with_suffix('.provenance.json').read_text()),
        retarget_script_sha256=hashlib.sha256((ROOT/'scripts/prepare_t800.py').read_bytes()).hexdigest(),
        limitations=['This new walk1 crop is not the old walk2 crop. Comparing its result to old runs does not isolate cropping.',
          'Full-vs-crop changes the training-state distribution and boundary frequency together; it does not separately estimate each mechanism.'])
    destination.with_suffix('.provenance.json').write_text(json.dumps(result,indent=2)+'\n')
    (ROOT/'reports/task_alignment_study/data-protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
