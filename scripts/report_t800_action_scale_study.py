"""Produce a reproducible HTML/JSON report for the completed short-motion ablation."""
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/t800_action_scale_study"


def read(path):
    return json.loads(path.read_text())


def table(headers, rows):
    row = lambda values, tag: "<tr>" + "".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in values) + "</tr>"
    return "<table>" + row(headers, "th") + "".join(row(values, "td") for values in rows) + "</table>"


def report_user_stop(status,design,gate):
    stop=read(OUT/"user-stop-record.json")
    interim=read(OUT/"interim_4000/comparison.json")
    result=dict(status="stopped_by_user",scope="2000-update screen and 4000-update comparison completed; planned 8000-update treatment comparison not completed",
                last_recorded_policy_updates=stop["runtime_before"].get("updates"),saved_checkpoints=stop["saved_checkpoints"],
                design=design,screening_gate=gate,interim_4000=interim,user_stop=stop,
                reward_frame_audit=read(OUT/"reward-frame-audit.json"))
    (OUT/"delivery.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    a,b=interim["control"],interim["treatment"]
    rows=[[label,f"{a[key]:.4f}",f"{b[key]:.4f}"] for key,label in
          (("normalized_return","评估 normalized return"),("joint_position_rmse_rad","关节 RMSE (rad)"),
           ("mean_root_position_error_m","根位置平均误差 (m)"),("mean_abs_yaw_error_deg","平均 yaw 误差 (°)"))]
    page="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>T800 动作尺度实验：按用户要求停止</title><style>body{font:16px/1.7 system-ui;max-width:1000px;margin:32px auto;padding:0 20px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccd5dc;padding:8px;text-align:left}th{background:#edf4f8}</style><body><h1>T800 动作尺度短动作实验</h1>"
    page+=f"<p>已按用户要求停止训练。停止前最后记录 {result['last_recorded_policy_updates']} 次策略更新；可续训的最新已保存检查点为 4768 次，另保留 4000 次里程碑。8000 次新尺度验证未完成，不能报告其结果。</p>"
    page+="<p>4000 次同预算比较：新尺度的关节误差小幅改善，回报基本相同。局部姿态与速度仍是主要丢分项；未证明动作尺度是主要根因。</p>"+table(["指标","原固定尺度 / 4000","新计算尺度 / 4000"],rows)
    page+="<p><a href='interim_4000/report.html'>4000 次详细评估与上游口径说明</a> · <a href='screening-2000-report.html'>2000 次原始筛查</a> · <a href='delivery.json'>完整保留结果</a> · <a href='user-stop-record.json'>停止与检查点记录</a> · <a href='reward-frame-audit.json'>Anchor 身体离线审计</a></p></body></html>"
    (OUT/"report.html").write_text(page)
    print(json.dumps({"status":"stopped_by_user","last_recorded_updates":result["last_recorded_policy_updates"],"saved_checkpoints":stop["saved_checkpoints"]},ensure_ascii=False))


def record(path, robust):
    f = read(path)
    tr = np.load(path.with_suffix(".trajectory.npz"))
    errors = tr["qpos"][:, 7:] - tr["reference_qpos"][:, 7:]
    joints = dict(zip(f["action_parameterization"]["joint_names"], np.sqrt(np.mean(errors**2, axis=0)).astype(float)))
    assert np.isclose(np.sqrt(np.mean(errors**2)), f["joint_position_rmse_rad"], atol=1e-6)
    return dict(report=str(path.relative_to(ROOT)), final=f, per_joint_rmse_rad=joints,
                leg_rmse_rad=float(np.sqrt(np.mean(errors[:,:12]**2))),
                arm_rmse_rad=float(np.sqrt(np.mean(errors[:,13:23]**2))),
                reset_completed=sum(r["completed_clip"] for r in robust), reset_count=len(robust),
                reset_mean_joint_rmse_rad=float(np.mean([r["joint_position_rmse_rad"] for r in robust])),
                reset_mean_full_clip_return=float(np.mean([r["normalized_full_clip_return"] for r in robust])),
                reset_std_full_clip_return=float(np.std([r["normalized_full_clip_return"] for r in robust])),
                reset_episodes=robust)


def main():
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    status, design, gate = [read(OUT / f"{name}.json") for name in ("status", "design", "gate-result")]
    if status["status"]=="cancelled_by_user":
        report_user_stop(status,design,gate)
        return
    assert status["status"] in ("evaluations_complete", "complete"), status
    budgets = [2000, 8000] if "t800_short_torque_kp_8000" in status["completed"] else [2000]
    extension = read(OUT/"exploratory-extension.json") if (OUT/"exploratory-extension.json").exists() else None
    comparisons, checks = {}, {}
    for budget in budgets:
        folder = ROOT / f"reports/training/t800_short_torque_kp_{budget}"
        assert read(folder / "status.json")["status"] == "finished"
        runtime = read(ROOT / f"reports/microbatch/t800_short_torque_kp_{budget}/result.json")
        assert runtime["status"] == "passed" and runtime["updates"] == budget
        update_rows = [json.loads(line) for line in (ROOT / f"reports/microbatch/t800_short_torque_kp_{budget}/updates.jsonl").read_text().splitlines()]
        assert len(update_rows) == (2000 if budget == 2000 else 6000)
        assert all(np.isfinite(v) for r in update_rows for key in ("metrics", "world_model_metrics") for v in r[key].values())
        expected = "initial_network_equality" if budget == 2000 else "resume_exact_state_checks"
        assert all(runtime[expected].values()), runtime[expected]
        checks[budget] = dict(runtime=runtime, all_update_metrics_finite=True)
        control = record(OUT / f"control_{budget}.json", [read(OUT / f"control_{budget}_reset_seed_{seed}.json") for seed in range(1,6)])
        treatment = record(folder / "final.json", [r for r in read(folder / "robustness.json") if r["checkpoint_updates"] == budget])
        assert control["reset_count"] == treatment["reset_count"] == 5
        a,b = control["final"],treatment["final"]
        comparisons[budget] = dict(control=control, treatment=treatment,
            joint_rmse_reduction_fraction=1-b["joint_position_rmse_rad"]/a["joint_position_rmse_rad"],
            full_clip_return_delta=b["normalized_full_clip_return"]-a["normalized_full_clip_return"],
            body_orientation_reward_delta=b["reward_fraction_of_term_max"]["motion_body_ori"]-a["reward_fraction_of_term_max"]["motion_body_ori"])

    last = comparisons[budgets[-1]]
    improvement = last["joint_rmse_reduction_fraction"]
    if 8000 in budgets and improvement >= .1 and last["full_clip_return_delta"] >= 0:
        conclusion = "短动作同预算结果支持动作参数化与既有优化配置不匹配是当前误差的重要原因；仅替换动作尺度就改善了跟踪。"
    elif 8000 in budgets:
        conclusion = "动作尺度变化在 2000 次时改善了结果，但 8000 次同预算下的优势有限或存在回报权衡；不能据此认定它是主要误差来源。"
    elif improvement > 0 and last["full_clip_return_delta"] > 0:
        conclusion = (f"动作尺度是一个有影响的因素：2000 次同预算下关节 RMSE 下降 {100*improvement:.1f}%，完整回报提高。"
                      "但未达到预设的 10% 续训条件，且主要姿态残差仍在；不能据此认定原固定尺度是主要原因，也尚无新尺度的 8000 次结果。")
    else:
        conclusion = "2000 次筛查未通过预设续训条件，未自动扩展到 8000 次。本实验尚不支持直接采用该尺度规则，也不能排除其他尺度或更长训练的收益。"
    limitations = ["只有一个训练种子（0）；五个扰动种子检验回放鲁棒性，不构成五次独立训练。",
                   "唯一配置干预是 action scale，但它同时改变同一原始动作下的 PD 目标、物理探索噪声、梯度尺度和 action-rate 正则的物理意义；尚未分离这些机制。",
                   "单步世界模型误差较小不等于 H32 闭环预测、离策略梯度和价值函数完全准确。",
                   "固定世界模型窗口的动作已重编码，使物理目标一致；不同组当期轨迹上的误差存在状态分布差异。",
                   "该动作约 4 秒，来自同一条 T800 重定向动作；结论不能直接外推到长动作、其他机器人或多动作任务。",
                   "上游未提供同评估协议的关节 RMSE；本实验检验本地因果假设，不宣称复现了上游全部指标。"]
    summary = dict(status="complete", generated_utc=datetime.now(timezone.utc).isoformat(), conclusion=conclusion,
                   new_policy_updates=budgets[-1], design=design, gate=gate, exploratory_extension=extension, comparisons=comparisons,
                   validation=checks, limitations=limitations)
    frame_audit=read(OUT/"reward-frame-audit.json") if (OUT/"reward-frame-audit.json").exists() else None
    summary["reward_frame_audit"]=frame_audit
    (OUT / "delivery.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n")
    nominal, robust, rewards, wm = [], [], [], []
    for budget, comparison in comparisons.items():
        for group, label in (("control", "RL Lab 固定尺度"), ("treatment", "0.25 × 力矩上限 / Kp")):
            r = comparison[group]
            f = r["final"]
            nominal.append([budget,label,f"{f['steps']}/{f['reference_frames']}",f"{f['normalized_full_clip_return']:.4f}",
                            f"{f['joint_position_rmse_rad']:.4f}",f"{100*f['mean_root_position_error_m']:.2f}",f"{f['mean_abs_yaw_error_deg']:.2f}"])
            robust.append([budget,label,f"{r['reset_completed']}/5",f"{r['reset_mean_joint_rmse_rad']:.4f}",
                           f"{r['reset_mean_full_clip_return']:.4f} ± {r['reset_std_full_clip_return']:.4f}"])
            rewards.append([budget,label]+[f"{100*f['reward_fraction_of_term_max'][k]:.2f}%" for k in
                            ("motion_body_pos","motion_body_ori","motion_body_lin_vel","motion_body_ang_vel")]+
                            [f"{f['mean_step_reward_terms']['action_rate_l2']:.6f}",f"{100*f['torque_saturation_fraction']:.3f}%"])
            wm.append([budget,label]+[f"{f[key][h-1]:.5f}" if f.get(key) and len(f[key])>=h else "无有效窗口"
                       for key in ("world_model_eval_nse_by_horizon","world_model_fixed_test_nse_by_horizon") for h in (1,8,32)])
    names = design["treatment_action_parameterization"]["joint_names"]
    joint_rows = []
    for name in names:
        joint_rows.append([name, f"{design['control_action_parameterization']['scale'][name]:.3f}",
                           f"{design['treatment_action_parameterization']['scale'][name]:.3f}"]+
                          [f"{comparison[group]['per_joint_rmse_rad'][name]:.4f}" for comparison in comparisons.values() for group in ("control","treatment")])
    fig, axes = plt.subplots(2, 2, figsize=(13,9), constrained_layout=True)
    idx = np.arange(len(names))
    for group, label, shift in (("control","RL Lab",-.2),("treatment","Torque/Kp",.2)):
        r = last[group]
        axes[0,0].bar(idx+shift, [r["per_joint_rmse_rad"][n] for n in names], width=.4, label=label)
        axes[0,1].bar(idx+shift,r["final"]["physical_target_offset_rms_per_joint_rad"],width=.4,label=label)
        tr = np.load((ROOT/r["report"]).with_suffix(".trajectory.npz"))
        for ax, name in zip(axes[1], ("J16_ELBOW_PITCH_L","J21_ELBOW_PITCH_R")):
            j = names.index(name)
            if group == "control":
                ax.plot(tr["time_s"],tr["reference_qpos"][:,7+j],"k--",label="Reference")
            ax.plot(tr["time_s"],tr["qpos"][:,7+j],label=label)
            ax.set(title=name,xlabel="Time (s)",ylabel="Joint angle (rad)")
    for ax,title in zip(axes[0],("Joint RMSE (rad)","Physical PD target offset RMS (rad)")):
        ax.set(title=title,xticks=idx,xticklabels=[n[:3] for n in names])
        ax.tick_params(axis="x",rotation=90)
    for ax in axes.flat:
        ax.legend()
        ax.grid(alpha=.2)
    fig.suptitle(f"T800 short motion, {budgets[-1]} policy updates, seed 0")
    fig.savefig(OUT/"joint-comparison.png",dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    expression = re.compile(r"updates (\d+) \| train_ret ([-\d.]+).*?wm nse1 ([\d.]+) nse8 ([\d.]+)")
    for label, runs in (("RL Lab",["t800_walk2_policy_ieee_stage1"]+(["t800_short_noanchor_8000"] if 8000 in budgets else [])),
                        ("Torque/Kp",[f"t800_short_torque_kp_{b}" for b in budgets])):
        values=[]
        for run_name in runs:
            log=ROOT/f"reports/training/{run_name}/training.log"
            if log.exists():
                values.extend(tuple(map(float,m.groups())) for m in expression.finditer(log.read_text()))
        if values:
            values=np.asarray(values)
            for i,ax in enumerate(axes):
                ax.plot(values[:,0],values[:,i+1],label=label,alpha=.8)
    for ax,title in zip(axes,("Training episodic return","Training WM NSE H1","Training WM NSE H8")):
        ax.set(title=title,xlabel="Policy updates")
        ax.grid(alpha=.2)
        ax.legend()
    fig.savefig(OUT/"training-curves.png",dpi=150)
    plt.close(fig)
    if 8000 in budgets:
        fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
        for group,label,run_name in (("control","RL Lab","t800_short_noanchor_8000"),
                                      ("treatment","Torque/Kp","t800_short_torque_kp_8000")):
            milestones={r["checkpoint_updates"]:r for r in read(ROOT/f"reports/training/{run_name}/evaluations.json") if r["checkpoint_updates"]>=2000}
            for b in budgets:
                milestones[b]=comparisons[b][group]["final"]
            points=sorted(milestones)
            for ax,key,title in zip(axes.flat,("normalized_full_clip_return","joint_position_rmse_rad","mean_root_position_error_m","motion_body_ori"),
                                   ("Full-clip return","Joint RMSE (rad)","Root position error (m)","Local orientation reward / max")):
                values=[milestones[b]["reward_fraction_of_term_max"][key] if key=="motion_body_ori" else milestones[b][key] for b in points]
                ax.plot(points,values,"o-",label=label)
                ax.set(title=title,xlabel="Policy updates")
                ax.grid(alpha=.2)
                ax.legend()
        fig.savefig(OUT/"evaluation-curves.png",dpi=150)
        plt.close(fig)
    css="body{font:16px/1.7 system-ui;margin:36px auto;max-width:1200px;padding:0 24px;color:#16232d}table{border-collapse:collapse;width:100%;margin:18px 0;font-size:14px}th,td{padding:8px;border:1px solid #ccd5dc;text-align:left}th{background:#edf4f8}img{max-width:100%}code{background:#eef2f5;padding:2px 5px}"
    page=f"<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>T800 动作尺度短动作对照</title><style>{css}</style><body><h1>T800 动作尺度短动作对照</h1><p>{html.escape(conclusion)}</p>"
    page+=f"<p>同一动作、seed 0、无 anchor、相同初始网络；4096 环境、4096 策略批量、H32、UTD 0.4、IEEE FP32、policy recomputation、原始学习率时间表。Kp/Kd、力矩限幅、默认姿态、奖励权重、原始动作噪声标准差均保持一致。新训练共 {budgets[-1]} 次策略更新。</p>"
    page+="<h2>完整动作回放</h2>"+table(["更新次数","动作尺度","完成帧数","完整回报","关节 RMSE (rad)","根位置误差 (cm)","yaw 误差 (°)"],nominal)
    if 8000 in budgets:
        page+="<img src='evaluation-curves.png'>"
    page+="<h2>五组初始扰动</h2>"+table(["更新次数","动作尺度","完成数","平均关节 RMSE (rad)","完整回报均值 ± 标准差"],robust)
    page+="<h2>奖励分解</h2>"+table(["更新次数","动作尺度","局部位置","局部朝向","线速度","角速度","动作变化项/步","力矩饱和率"],rewards)
    page+="<h2>世界模型 NSE</h2><p>先列各组当期轨迹的 H1/H8/H32，再列相同物理测试窗口的 H1/H8/H32。</p>"+table(["更新次数","动作尺度","当期 H1","当期 H8","当期 H32","固定 H1","固定 H8","固定 H32"],wm)
    page+="<h2>训练过程</h2><p>训练回报来自随机起点、有噪声的训练回合，与上面的标准完整动作回放不同。世界模型训练误差也取决于各组采集到的状态分布。</p><img src='training-curves.png'>"
    page+="<h2>关节分析</h2><img src='joint-comparison.png'>"+table(["关节","原尺度","新尺度"]+[f"{b} {'原' if g=='control' else '新'} RMSE" for b in budgets for g in ("control","treatment")],joint_rows)
    page+="<h2>动作尺度与实际输出</h2><p>尺度增大允许较小的原始动作产生相同的物理目标，但策略仍可主动缩小原始输出。下面比较最终预算下的动作与误差，不能仅依据尺度倍数推断跟踪幅度已改善。</p>"
    action_rows=[]
    for joint in ("J15_SHOULDER_YAW_L","J16_ELBOW_PITCH_L","J17_ELBOW_YAW_L","J20_SHOULDER_YAW_R","J21_ELBOW_PITCH_R","J22_ELBOW_YAW_R"):
        j=names.index(joint)
        action_rows.append([joint]+[f"{last[g]['final'][field][j]:.4f}" for field in
                            ("raw_action_rms_per_joint","physical_target_offset_rms_per_joint_rad") for g in ("control","treatment")])
    page+=table(["关节","原尺度动作 RMS","新尺度动作 RMS","原物理目标偏移 RMS (rad)","新物理目标偏移 RMS (rad)"],action_rows)
    page+="<p>原尺度已有的单步世界模型与对角动作梯度验证不能排除多步预测、离策略动作或价值估计的误差；动作参数化的实验收益也不能单独证明具体是哪一种优化机制在起作用。</p>"
    if frame_audit:
        frame_rows=[]
        for name,run in frame_audit["runs"].items():
            frame_rows.append([name]+[f"{100*run['variants'][v]['mean_pose_reward_fraction']['local_body_orientation']:.2f}%" for v in
                                      ("base_15_current","base_14_no_head","waist_15","waist_14_no_head")])
        page+="<h2>仍存在的适配语义差异</h2><p>G1 使用 torso_link 作为 anchor、跟踪 14 个身体；T800 当前使用 LINK_BASE、跟踪 15 个身体（包含头部）。即使额外 anchor observation 关闭，anchor 仍用于奖励对齐。本次尺度对照未更改这些定义。</p>"
        page+="<p>以下仅对同一份动作前状态和配对参考姿态离线重新评分，列出局部朝向奖励比例。它不是环境动作后回报，也不代表重新训练的效果；LINK_WAIST_YAW 只是待验证的躯干对应候选。</p>"
        page+=table(["轨迹","基座 / 15 身体","基座 / 去掉头部","腰部躯干 / 15 身体","腰部躯干 / 去掉头部"],frame_rows)
        page+="<p>其他差异包括 IEEE FP32 矩阵乘法、策略重计算及软件栈；长期训练数值等价性尚未证明。T800 动力学和重定向动作也构成不同任务，不能把跨机器人的公开回报或缺失的 RMSE 当作严格同任务验收值。<a href='reward-frame-audit.json'>离线评分与姿态梯度记录</a>。</p>"
    page+="<h2>预设条件与验证</h2><p>2000 次后要求关节 RMSE 至少下降 10%、完整回报和局部朝向奖励不降低、标准回放完成、五组扰动至少完成四组。实际条件：</p><pre>"+html.escape(json.dumps(gate["checks"],indent=2,ensure_ascii=False))+"</pre>"
    if extension:
        page+="<p><strong>追加验证说明：</strong>2000 次下降 8.5%，未达到原定 10% 门槛。鉴于回报及扰动结果同步改善，且新组早期收敛较慢，随后决定补齐 8000 次同预算验证。这是看到筛查结果后作出的追加决定；原门槛与未通过记录完整保留，8000 次结果属于探索性验证。<a href='exploratory-extension.json'>追加决定记录</a> · <a href='screening-2000-delivery.json'>原始筛查报告</a>。</p>"
    page+="<p>GPU 控制目标/力矩/回放检查、配置一致性、错误尺度续训拒绝、初始网络逐张量内容摘要相等均通过。新检查点包含尺度元数据，评估自动加载对应映射；源算法、旧训练入口和已有结果保留。</p><h2>解释边界</h2><ul>"+"".join(f"<li>{html.escape(x)}</li>" for x in limitations)+"</ul>"
    page+="<p><a href='delivery.json'>完整机器可读结果</a> · <a href='design.json'>预先固定的设计</a> · <a href='gate-result.json'>续训判定</a> · <a href='control-chain.json'>GPU 控制链验证</a></p></body></html>"
    command=("/home/ethan/miniconda3/envs/supermimic/bin/python scripts/evaluate_action_scale.py --robot t800 "
             f"--checkpoint logs/t800_short_torque_kp_{budgets[-1]}/ckpt_latest.pt "
             "--motion data/npz/t800_single/walk2_steady.npz --wm-horizon 32 --out /tmp/t800-scale-eval.json")
    page=page.replace("</body></html>","<h2>检查点使用</h2><p>新尺度检查点使用这个评估入口；它会根据检查点元数据恢复动作映射。旧的通用入口尚不读取该实验元数据。</p><pre style='white-space:pre-wrap'>"+html.escape(command)+"</pre></body></html>")
    (OUT/"report.html").write_text(page)
    status.update(status="complete",delivery="reports/t800_action_scale_study/report.html",conclusion=conclusion)
    (OUT/"status.json").write_text(json.dumps(status,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({"conclusion":conclusion,"budget":budgets[-1],"joint_rmse_reduction_fraction":improvement},ensure_ascii=False))


if __name__ == "__main__":
    main()
