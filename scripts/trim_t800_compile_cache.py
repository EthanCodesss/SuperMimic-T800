"""Reclaim only old regenerable Triton caches after the G1 process has exited."""
import json
import os
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/t800_fkfix_study'


def main():
    plan=json.loads((OUT/'handoff-plan.json').read_text())
    proc=Path(f"/proc/{plan['g1_training_pid']}/stat")
    assert not proc.exists() or proc.read_text().split(') ',1)[1][0]=='Z', 'G1 must have released the GPU first'
    cutoff=(ROOT/'logs/g1_walk1_e1c_fkfixed_8104/ckpt_initial.pt').stat().st_mtime-60
    cache=Path('/tmp/torchinductor_ethan/triton')
    removed=[]
    if cache.is_dir() and not cache.is_symlink():
        assert cache.stat().st_uid==os.getuid()
        for p in sorted(cache.iterdir(),key=lambda p:p.stat().st_mtime):
            if shutil.disk_usage(ROOT).free>=4*2**30:break
            if not p.is_dir() or p.is_symlink():continue
            files=[f for f in p.rglob('*') if f.is_file() and not f.is_symlink()]
            if not files or max(f.stat().st_mtime for f in files)>=cutoff:continue
            removed.append(dict(path=str(p),bytes=sum(f.stat().st_size for f in files)))
            shutil.rmtree(p)
    report=dict(removed=removed,free_gib=shutil.disk_usage(ROOT).free/2**30,
        scope='Only Triton cache entries predating the current G1 run. No model, dataset, report, or checkpoint removed.')
    (OUT/'cache-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(removed_directories=len(removed),free_gib=report['free_gib'])))


if __name__=='__main__':main()
