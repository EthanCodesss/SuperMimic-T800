"""Freeze the T800 protocol only after full-batch and evaluator preflight pass."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/t800_fkfix_study'
RUN='t800_walk1_full_fkfixed_8104'
PROBE='t800_fkfixed_fullbatch_policy_probe'


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    from scripts.train_strict import verify_sources
    verify_sources()
    probe=read(ROOT/'reports/microbatch'/PROBE/'result.json')
    assert probe['status']=='passed' and probe['additional_updates']==3
    assert probe['execution']['effective_policy_batch']==4096 and probe['execution']['horizon']==32
    audit=read(OUT/'profile-audit.json');assert audit['status']=='passed'
    ev=read(OUT/'probe-evaluation.json')
    benchmark=read(OUT/'probe-benchmark.json')
    assert ev['checkpoint_updates']==benchmark['checkpoint_updates']==3
    ex=read(ROOT/'reports/microbatch'/RUN/'execution.json')
    cfg=read(ROOT/'reports/microbatch'/RUN/'config.json')
    pcfg=read(ROOT/'reports/microbatch'/PROBE/'config.json')
    keys=['env','world_model','wm_trainer','policy','critic','policy_trainer','schedule','seed']
    assert all(cfg[k]==pcfg[k] for k in keys)
    for key in ['action_parameterization','task_adapter','fk_compile_compat','t800_profile_sha256']:
        assert ex[key]==probe['execution'][key],key
    assert ev['fk_compile_compat']==ex['fk_compile_compat']
    assert cfg['policy']['goal_dim']==600 and cfg['env']['anchor_obs'] is False
    initial=ROOT/'logs'/PROBE/'ckpt_initial.pt'
    assert initial.is_file()
    old=read(ROOT/'reports/task_alignment_study/t800-full-protocol.json')
    old.update(created_utc=datetime.now(timezone.utc).isoformat(),run_name=RUN,
        reference_initial=str(initial.relative_to(ROOT)),
        fk_compile_compat=ex['fk_compile_compat'],t800_profile_sha256=ex['t800_profile_sha256'],
        task_adapter=ex['task_adapter'],action_parameterization=ex['action_parameterization'],
        initialization='random_seed_0',
        preflight=dict(full_batch_result=f'reports/microbatch/{PROBE}/result.json',
            combined_profile_audit='reports/t800_fkfix_study/profile-audit.json',
            physical_evaluator='reports/t800_fkfix_study/probe-evaluation.json',
            benchmark='reports/t800_fkfix_study/probe-benchmark.json'),
        comparison_limit='This is a new corrected T800 baseline with full walk1, not a one-variable comparison against historical short walk2 runs.')
    old['evaluation']=dict(checkpoint_interval_control_steps=1000,checkpoint_interval_updates=0,
        checkpoint_updates=[2000,4052,8104],fixed_wm_windows=None,
        robustness_seeds_final=[1,2,3,4,5],deterministic_full_clip=True,
        physical_evaluator='scripts/evaluate_t800_fk_compat.py',benchmark_evaluator='scripts/evaluate_t800_fk_benchmark.py',
        wm_test_window_rule='Freeze the first valid independent evaluation windows for later checkpoints',
        checkpoint_metrics=['normalized_return','normalized_full_clip_return','joint_position_rmse_rad','mean_root_position_error_m','mean_abs_yaw_error_deg','mean_step_reward_terms'])
    files=['scripts/train_t800_fk_compat.py','scripts/run_t800_fk_compat_stage.py','scripts/evaluate_t800_fk_compat.py',
        'scripts/evaluate_t800_fk_benchmark.py','scripts/t800_fk_profile.py','scripts/fk_compile_compat.py',
        'scripts/t800_task_adapter.py','scripts/t800_action_scaling.py','scripts/train_strict.py',
        'configs/strict-source-lock.json',old['motion']['path'],old['reference_initial'],
        'reports/task_alignment_study/data-protocol.json','reports/task_alignment_study/reference-quality-status.json',
        'reports/t800_fkfix_study/profile-audit.json']
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'optimizations').glob('*.py')))
    old['frozen_file_sha256']={p:sha(ROOT/p) for p in files}
    path=OUT/'protocol.json'
    assert not path.exists(), 'Preserve any existing frozen protocol'
    path.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='passed',protocol=str(path),initial_sha256=sha(initial))))


if __name__=='__main__':main()
