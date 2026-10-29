"""User-requested live checkpoint review; suspend and resume the same trainer."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/t800_action_scale_study/interim_4000"


def main():
    os.chdir(ROOT)
    state=json.loads((ROOT/"reports/training/t800_short_torque_kp_8000/status.json").read_text())
    pid=state["training_pid"]
    command=Path(f"/proc/{pid}/cmdline").read_text().replace("\x00"," ")
    assert "train_action_scale.py" in command and "t800_short_torque_kp_8000" in command,command
    OUT.mkdir(parents=True,exist_ok=False)
    status={"status":"preparing","training_pid":pid,"command":command,"started_unix":time.time(),
            "scope":"Immutable 4000-update checkpoints; trainer temporarily SIGSTOP'd, no checkpoint resume or RNG/replay reset"}
    def write():
        (OUT/"pause-and-evaluation.json").write_text(json.dumps(status,indent=2)+"\n")
    write()
    # A separate process resumes the trainer if this wrapper disappears or hangs.
    watchdog_code="""import os,sys,time,signal
from pathlib import Path
parent,trainer=map(int,sys.argv[1:])
deadline=time.monotonic()+600
while time.monotonic()<deadline and Path(f'/proc/{parent}').exists(): time.sleep(1)
p=Path(f'/proc/{trainer}/cmdline')
if p.exists() and b't800_short_torque_kp_8000' in p.read_bytes(): os.kill(trainer,signal.SIGCONT)
"""
    watcher=subprocess.Popen([sys.executable,"-c",watchdog_code,str(os.getpid()),str(pid)],start_new_session=True)
    def interrupted(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,interrupted)
    try:
        os.kill(pid,signal.SIGSTOP)
        time.sleep(2)
        assert "\nState:\tT" in Path(f"/proc/{pid}/status").read_text()
        status["status"]="evaluating_trainer_suspended"
        write()
        env=os.environ.copy()
        env.update(MPLCONFIGDIR=str(ROOT/".cache/matplotlib"),OMP_NUM_THREADS="4")
        jobs=[("control", "logs/t800_short_noanchor_8000/ckpt_u0004000.pt",0,
               "reports/training/t800_walk2_policy_ieee_stage1/baseline.wm_windows.pt")]
        jobs += [("treatment"+(f"_reset_seed_{seed}" if seed else ""),
                  "logs/t800_short_torque_kp_8000/ckpt_u0004000.pt",seed,
                  "reports/t800_action_scale_study/fixed_torque_kp.wm_windows.pt") for seed in range(6)]
        for label,checkpoint,seed,windows in jobs:
            cmd=[sys.executable,"scripts/evaluate_action_scale.py","--robot","t800","--checkpoint",checkpoint,
                 "--motion","data/npz/t800_single/walk2_steady.npz","--out",str(OUT/f"{label}.json"),
                 "--wm-horizon","32","--wm-test-windows",windows,"--seed",str(seed)]
            if seed:
                cmd.append("--reset-noise")
            with (OUT/f"{label}.log").open("w") as log:
                subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,check=True)
            f=json.loads((OUT/f"{label}.json").read_text())
            print(label,{k:f[k] for k in ("checkpoint_updates","completed_clip","normalized_full_clip_return","joint_position_rmse_rad")},flush=True)
        status["status"]="passed"
    except BaseException as exc:
        status.update(status="failed",error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        os.kill(pid,signal.SIGCONT)
        status.update(trainer_resumed=True,finished_unix=time.time())
        write()
        watcher.terminate()
        watcher.wait(timeout=5)


if __name__=="__main__":
    main()
