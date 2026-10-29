"""Summarize matched standing experiments and control-chain checks."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def read(path):
    return json.loads(Path(path).read_text())


def main():
    rows=[]
    diagnostics=[]
    for robot in ("g1", "t800"):
        for mode, suffix in (("initial", "initial"), ("critic", "critic-2000"), ("no_critic", "no-critic-2000")):
            path=f"reports/{robot}-standing-{suffix}.json"
            r=read(path)
            rows.append({"robot":robot, "mode":mode, "evaluation":path,
                "steps":r["steps"], "duration_s":r["steps"]*.02,
                "completed_clip":r["completed_clip"], "normalized_return":r["normalized_return"],
                "root_error_m":r["mean_root_position_error_m"], "joint_rmse_rad":r["joint_position_rmse_rad"],
                "hard_joint_limit_fraction":r["hard_joint_limit_fraction"],
                "torque_saturation_fraction":r["torque_saturation_fraction"]})
            if mode != "initial":
                run=f"{robot}_standing_{'label_fix' if mode=='critic' else 'no_critic'}"
                e=EventAccumulator(f"logs/{run}/tb").Reload()
                g=e.Scalars("policy/policy_grad_norm")
                bad=[v.step for v in g if not math.isfinite(v.value)]
                diagnostics.append({"run":run,"first_logged_nonfinite_policy_grad_norm":bad[0] if bad else None,
                    "updates":read(f"logs/{run}/run_summary.json")["updates"],
                    "guard_enabled_during_run":False})
    result={"date":"2026-10-27", "status":"control_and_observation_parity_passed; replay_label_bug_fixed; standing_learning_unresolved",
        "confirmed_bug":{
            "location":"src/supermimic/data/replay_buffer.py:sample_windows",
            "before":"Read done from next row, though recorder stores done(s_next) with the current action; penultimate transition labelled terminal early.",
            "after":"Read done from the same row as action and reward.",
            "regression":"tests/test_replay_labels.py fails before the fix and passes after it.",
            "upstream_snapshot":"Unmodified; fix only in working src."},
        "control_checks":{r:read(f"reports/{r}-control-chain.json") for r in ("g1","t800")},
        "input_parity":"Both robots pass full 10-frame observation reconstruction and future+anchor goal checks against physical environment inputs.",
        "experiment":{"iterations":2000,"parallel_envs":128,"seed":0,"reference_duration_s":6.,
            "reference":"Each robot's default joint pose, root adjusted to 2mm foot clearance; synthetic, not mocap.",
            "horizon":32,"model_fit_window":8,"anchor_observations":True,
            "wm_hidden":[512,512,512],"policy_hidden":[512,256,128],
            "label_fix_in_all_runs":True,"ablation":"Only use_critic differs within each robot; same seed, but subsequent RNG consumption can differ.",
            "evaluations_per_checkpoint":1},
        "evaluations":rows,"optimization_diagnostics":diagnostics,
        "fixed_model_errors":{r:read(f"reports/{r}-standing-fixed-model-errors.json") for r in ("g1","t800")},
        "model_action_response":"reports/t800-walk-model-response.json",
        "interpretation":[
            "No joint-order, action-scale, default-offset or instantaneous PD force mismatch found in tested probes.",
            "The replay-label fix alone did not establish stable standing.",
            "G1 also degrades on a static reference, so failure is not exclusive to T800 or the LAFAN1 walking motion.",
            "Disabling critic avoided the observed G1 numerical blow-up in this one-seed comparison and modestly increased survival for both robots; neither completes the clip and returns remain below their initial baselines.",
            "Eight-step model fit does not establish 32-step closed-loop reliability; critic gradients and model rollout stability remain priorities.",
            "These tests do not prove contact-force accuracy, walking dynamic feasibility or an inherent defect in the upstream algorithm."],
        "post_experiment_guard":"Actor, critic and world-model gradient clipping now raises on a nonfinite total gradient norm before applying that optimizer step; this is detection, not a convergence fix.",
        "guard_smoke":read("logs/t800_guard_smoke/run_summary.json"),
        "source_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path("src").rglob("*.py"))},
        "artifacts":{"plot":"reports/control-diagnosis.png","plot_pdf":"reports/control-diagnosis.pdf"}}
    Path("reports/control-diagnosis.json").write_text(json.dumps(result,indent=2)+"\n")

    fig, ax=plt.subplots(1,3,figsize=(15,4.5))
    colors=["#94a3b8", "#dc685d", "#208f86"]
    modes=["initial","critic","no_critic"]
    labels=["Initial", "Critic", "No critic"]
    for j,(mode,label,color) in enumerate(zip(modes,labels,colors)):
        vals=[next(r for r in rows if r["robot"]==robot and r["mode"]==mode)["duration_s"] for robot in ("g1","t800")]
        bars=ax[0].bar(np.arange(2)+(j-1)*.24,vals,width=.22,color=color,label=label)
        ax[0].bar_label(bars,fmt="%.2f",fontsize=8)
    ax[0].axhline(6,ls="--",color="black",lw=1,label="6 s target")
    ax[0].set(xticks=[0,1],xticklabels=["G1","T800"],ylabel="Seconds until termination",ylim=(0,6.6),title="Physical standing evaluation")
    ax[0].legend(fontsize=8)
    for robot,color in (("g1","#586cbe"),("t800","#208f86")):
        for mode,style in (("label_fix","--"),("no_critic","-")):
            e=EventAccumulator(f"logs/{robot}_standing_{mode}/tb").Reload()
            vals=e.Scalars("train/episode_norm_return")
            ax[1].plot([v.step for v in vals],[v.value for v in vals],style,color=color,label=f"{robot.upper()} {'critic' if mode=='label_fix' else 'no critic'}",lw=1.4)
        for entry in result["fixed_model_errors"][robot]["results"][1:]:
            is_no="no_critic" in entry["checkpoint"]
            y=entry["nse_by_horizon"]
            ax[2].plot(np.arange(1,len(y)+1),y,"-" if is_no else "--",color=color)
    ax[1].set_yscale("symlog",linthresh=1)
    ax[1].set(xlabel="Environment steps",ylabel="Normalized training return",title="Training return (symlog)")
    ax[1].legend(fontsize=8)
    ax[2].set_yscale("log")
    ax[2].set(xlabel="Open-loop prediction steps",ylabel="Normalized squared state error",title="Fixed initial-policy windows")
    for a in ax: a.grid(axis="y",alpha=.2)
    fig.suptitle("Matched standing diagnostics · seed 0 · 128 environments · 2,000 steps",fontsize=12)
    fig.text(.5,.005,"One deterministic evaluation per checkpoint; survival is not tracking success. Dashed = critic; solid = no critic.",ha="center",fontsize=9)
    fig.tight_layout(rect=[0,.04,1,.95])
    for suffix in ("png","pdf"): fig.savefig(f"reports/control-diagnosis.{suffix}",dpi=170)
    plt.close(fig)
    print(json.dumps(rows,indent=2))


if __name__ == "__main__":
    main()
