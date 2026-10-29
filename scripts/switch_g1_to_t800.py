"""User-authorized handoff: preserve/evaluate G1 u4052, then validate/start T800."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/t800_fkfix_study'
G1 = 'g1_walk1_e1c_fkfixed_8104'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def alive(pid):
    p = Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split(') ', 1)[1][0] != 'Z'


def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    plan = read(OUT / 'handoff-plan.json')
    assert plan and plan['authorized_g1_checkpoint_updates'] == 4052
    pid, stage_pid = plan['g1_training_pid'], plan['g1_stage_pid']
    for process, expected in [(pid, 'scripts/train_memory_fk_compat.py'), (stage_pid, 'scripts/run_fk_compat_stage.py')]:
        cmd = Path(f'/proc/{process}/cmdline').read_bytes().replace(b'\0', b' ').decode()
        assert expected in cmd and G1 in cmd, (process, cmd)
    status = dict(status='waiting_g1_checkpoint', supervisor_pid=os.getpid(), plan=plan)
    def update(**values):
        status.update(values, updated_utc=datetime.now(timezone.utc).isoformat())
        write(OUT / 'handoff-status.json', status)
        print(json.dumps(values, ensure_ascii=False), flush=True)
    update()
    frozen = False
    try:
        checkpoint = ROOT / 'logs' / G1 / 'ckpt_u0004052.pt'
        while not checkpoint.exists():
            if not alive(pid):
                raise RuntimeError('G1 exited before publishing the required checkpoint')
            time.sleep(.05)
        # Atomic checkpoint publication precedes the next training update. Park
        # only the training process; the existing controller evaluates u4052.
        os.kill(pid, signal.SIGSTOP)
        frozen = True
        runtime = read(ROOT / 'reports/microbatch' / G1 / 'result.json')
        update(status='evaluating_g1_4052', preserved_checkpoint=str(checkpoint),
               parked_runtime_updates=(runtime or {}).get('updates'))
        folder = ROOT / 'reports/training' / G1
        physical_path = folder / 'eval_ckpt_u0004052.json'
        benchmark_path = folder / 'eval_ckpt_u0004052.upstream.json'
        while True:
            stage = read(folder / 'status.json') or {}
            physical, benchmark = read(physical_path), read(benchmark_path)
            if physical and benchmark and str(benchmark_path) in stage.get('upstream_benchmarks', []):
                break
            if not alive(stage_pid) or stage.get('status') in ['failed', 'interrupted']:
                raise RuntimeError('G1 evaluation controller failed before both u4052 results were recorded')
            time.sleep(2)
        assert physical['checkpoint_updates'] == benchmark['checkpoint_updates'] == 4052
        # SIGINT is handled by the original launcher. Its nonzero return is an
        # intentional user stop, not an optimizer failure. The u4052 file is kept.
        os.kill(pid, signal.SIGINT)
        os.kill(pid, signal.SIGCONT)
        frozen = False
        deadline = time.monotonic() + 180
        while alive(pid) or alive(stage_pid):
            if time.monotonic() > deadline:
                raise RuntimeError('G1 did not finish its orderly stop within 180 seconds')
            time.sleep(1)
        # Make the default resume name point at the exact preserved milestone,
        # rather than an older periodic checkpoint.
        latest_link=checkpoint.parent/f'.ckpt_latest.handoff.{os.getpid()}.link'
        os.link(checkpoint,latest_link)
        latest_link.replace(checkpoint.parent/'ckpt_latest.pt')
        stage = read(folder / 'status.json')
        stage['stop_controller_original_status'] = stage['status']
        if 'error' in stage:
            stage['stop_controller_original_error']=stage.pop('error')
        stage['status'] = 'paused'
        stage['pause'] = dict(reason='User requested pause after u4052 evaluation and switch to T800',
            checkpoint=str(checkpoint), checkpoint_updates=4052,
            physical_evaluation=str(physical_path), upstream_benchmark=str(benchmark_path),
            replay_rng_restored_on_future_resume=False)
        write(folder / 'status.json', stage)
        runtime_path = ROOT / 'reports/microbatch' / G1 / 'result.json'
        runtime = read(runtime_path)
        if 'error' in runtime:
            runtime['intentional_stop_detail']=runtime.pop('error')
        runtime.update(status='paused', phase='user_requested_pause', checkpoint=str(checkpoint),
                       checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                       last_process_reported_updates=runtime.get('updates'), updates=4052,
                       iteration=physical['checkpoint_iteration'],
                       paused_checkpoint_updates=4052, pause_reason=stage['pause']['reason'])
        write(runtime_path, runtime)
        update(status='waiting_t800_readiness', g1_physical=physical, g1_benchmark=benchmark)
        subprocess.run([sys.executable, 'scripts/report_g1_fkfix_study.py'], check=True)
        while not (ready := read(OUT / 'ready.json')):
            time.sleep(2)
        for name, digest in ready['frozen_files'].items():
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
        env = os.environ.copy()
        env.update(OMP_NUM_THREADS='4', MPLCONFIGDIR=str(ROOT / '.cache/matplotlib'), PYTHONUNBUFFERED='1')
        for task in ready['preflight']:
            update(status='t800_preflight', active_task=task['name'])
            with (OUT / task['log']).open('w') as log:
                result = subprocess.run(task['command'], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"Preflight {task['name']} failed; inspect {task['log']}")
            if task.get('assert_passed_report'):
                check = read(ROOT / task['assert_passed_report'])
                assert check and (check.get('passed') is True or check.get('status') == 'passed'), task['name']
        update(status='starting_t800', active_task=None)
        with (OUT / 'stage.log').open('x') as log:
            process = subprocess.Popen(ready['training_command'], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        update(status='t800_started', t800_stage_pid=process.pid, training_command=ready['training_command'])
        code = process.wait()
        update(status='t800_finished' if code == 0 else 'failed', t800_stage_returncode=code)
    except BaseException as exc:
        update(status='failed', error=f'{type(exc).__name__}: {exc}', g1_process_parked=frozen)
        # A parked process stays parked on a failed handoff: do not silently
        # continue a training run the user explicitly requested to pause.
        raise


if __name__ == '__main__':
    main()
