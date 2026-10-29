"""Freeze the T800 protocol only after full-batch and evaluator preflight pass."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'reports/t800_anchor_head_study'
RUN='t800_walk1_anchor_headfixed_8104'
PROBE='t800_anchor_head_fullbatch_probe'


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
    keys=['data','env','world_model','wm_trainer','policy','critic','policy_trainer','schedule','seed']
    assert all(cfg[k]==pcfg[k] for k in keys)
    for key in ['action_parameterization','task_adapter','fk_compile_compat','t800_profile_sha256','head_control']:
        assert ex[key]==probe['execution'][key],key
        assert ev[key]==ex[key],key
        assert benchmark['profile'][key]==ex[key],key
    assert ev['head_raw_action_max_abs']==0 and ev['anchor_obs'] is True
    assert ev['fk_compile_compat']==ex['fk_compile_compat']
    assert cfg['policy']['goal_dim']==609 and cfg['env']['anchor_obs'] is True
    baseline_cfg=read(ROOT/'reports/microbatch/t800_walk1_full_fkfixed_8104/config.json')
    expected=json.loads(json.dumps(baseline_cfg))
    expected['env']['anchor_obs']=True
    expected['policy']['goal_dim']=609
    expected['policy_trainer']['anchor_obs']=True
    for k in keys:
        assert cfg[k]==expected[k],f'Unexpected algorithm change: {k}'
    baseline_protocol=read(ROOT/'reports/t800_fkfix_study/protocol.json')
    assert ex['action_parameterization']==baseline_protocol['action_parameterization']
    assert ex['fk_compile_compat']==baseline_protocol['fk_compile_compat']
    task=dict(baseline_protocol['task_adapter'])
    task.update(anchor_obs=True,head_control=ex['task_adapter']['head_control'])
    assert task==ex['task_adapter']
    initial=ROOT/'logs'/PROBE/'ckpt_initial.pt'
    assert initial.is_file()
    old=read(ROOT/'reports/t800_fkfix_study/protocol.json')
    old.update(created_utc=datetime.now(timezone.utc).isoformat(),run_name=RUN,
        reference_initial=str(initial.relative_to(ROOT)),anchor_obs=True,head_control=ex['head_control'],
        fk_compile_compat=ex['fk_compile_compat'],t800_profile_sha256=ex['t800_profile_sha256'],
        task_adapter=ex['task_adapter'],action_parameterization=ex['action_parameterization'],
        initialization='random_seed_0',
        preflight=dict(full_batch_result=f'reports/microbatch/{PROBE}/result.json',
            combined_profile_audit='reports/t800_anchor_head_study/profile-audit.json',
            physical_evaluator='reports/t800_anchor_head_study/probe-evaluation.json',
            benchmark='reports/t800_anchor_head_study/probe-benchmark.json'),
        comparison_limit='User-requested combined anchor-observation and fixed-head intervention; cannot attribute gains independently to either change. Single seed; fresh seed 0 initialization with changed policy/critic input shapes, not identical baseline weights.')
    old['evaluation']=dict(checkpoint_interval_control_steps=1000,checkpoint_interval_updates=0,
        checkpoint_updates=[2000,4052,8104],fixed_wm_windows=None,
        robustness_seeds_final=[1,2,3,4,5],deterministic_full_clip=True,
        physical_evaluator='scripts/evaluate_t800_anchor_head.py',benchmark_evaluator='scripts/evaluate_t800_anchor_head_benchmark.py',
        wm_test_window_rule='Freeze the first valid independent evaluation windows for later checkpoints',
        checkpoint_metrics=['normalized_return','normalized_full_clip_return','joint_position_rmse_rad',
            'non_head_joint_rmse_rad','head_default_rmse_rad','head_raw_action_max_abs',
            'mean_root_position_error_m','mean_abs_yaw_error_deg','mean_step_reward_terms'])
    files=['scripts/train_t800_anchor_head.py','scripts/run_t800_anchor_head_stage.py','scripts/evaluate_t800_anchor_head.py',
        'scripts/evaluate_t800_anchor_head_benchmark.py','scripts/t800_anchor_head_profile.py','scripts/fk_compile_compat.py',
        'scripts/t800_task_adapter.py','scripts/t800_action_scaling.py','scripts/train_strict.py',
        'configs/strict-source-lock.json',old['motion']['path'],old['reference_initial'],
        'scripts/t800_fixed_head.py','scripts/t800_fk_profile.py','scripts/audit_t800_anchor_head.py',
        'scripts/prepare_t800_anchor_head_protocol.py','reports/t800_fkfix_study/protocol.json',
        'reports/task_alignment_study/data-protocol.json','reports/task_alignment_study/reference-quality-status.json',
        'reports/t800_anchor_head_study/profile-audit.json']
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT/'optimizations').glob('*.py')))
    old['comparison']=dict(baseline_run='t800_walk1_full_fkfixed_8104',
        changes=['9-D waist-anchor observations in real and imagined policy/critic goals',
                 'Fixed default head PD targets; masked head action outputs, execution noise and replay'],
        matched=['reference NPZ','25-D state/action schema','body reward mapping and weights','PD constants','action scales',
                 '4096 environments and actor batch','32-step imagination','8104 updates','learning schedule','seed 0'],
        unchanged_algorithm_sections_checked=keys,
        initialization_limit='Same random seed, different goal input sizes; baseline actor/critic weights are not identical.')
    old['frozen_file_sha256']={p:sha(ROOT/p) for p in files}
    path=OUT/'protocol.json'
    assert not path.exists(), 'Preserve any existing frozen protocol'
    path.write_text(json.dumps(old,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='passed',protocol=str(path),initial_sha256=sha(initial))))


if __name__=='__main__':main()
