"""Generate a local HTML delivery report and figures from the completed study."""
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/t800_anchor_study"
NAMES = ["t800_short_noanchor_8000", "t800_short_anchor_8000",
         "t800_long_noanchor_8000", "t800_long_anchor_8000"]
LABELS = ["短动作 / 无 anchor", "短动作 / anchor", "60 秒动作 / 无 anchor", "60 秒动作 / anchor"]


def read(path):
    return json.loads(path.read_text())


def number(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def table(headers, rows):
    return '<table><thead><tr>' + ''.join(f'<th>{html.escape(str(x))}</th>' for x in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{html.escape(str(x))}</td>' for x in row) + '</tr>' for row in rows) + '</tbody></table>'


def main():
    os.chdir(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    OUT.mkdir(parents=True, exist_ok=True)
    study_state = read(OUT / "status.json")
    cancelled = study_state.get("cancelled_stage")
    selected = [(name, label) for name, label in zip(NAMES, LABELS) if name != cancelled]
    scope_text = ("评估完善、短动作两组各累计 8000 次、长动作无 anchor 8000 次已完成。长动作 anchor 已按用户要求取消；取消前最后记录为 72 次更新，不计入正式比较。"
                  if cancelled else "四阶段均完成：评估完善、无 anchor 续训至 8000 次、anchor 同预算对照、连续长动作验证。")
    records = []
    for name, label in selected:
        folder = ROOT / "reports/training" / name
        final = read(folder / "final.json")
        assert final["checkpoint_updates"] == 8000
        status = read(folder / "status.json")
        assert status["status"] == "finished"
        evaluations = read(folder / "evaluations.json")
        unique = {r["checkpoint_updates"]: r for r in evaluations}
        if name == NAMES[0]:
            old = read(ROOT / "reports/training/t800_walk2_policy_ieee_stage1/evaluations.json")
            for r in old:
                if r["checkpoint_updates"] not in unique:
                    r["normalized_full_clip_return"] = r["normalized_return"] * r["steps"] / r["reference_frames"]
                    unique[r["checkpoint_updates"]] = r
        robustness = read(folder / "robustness.json")
        final_robust = [r for r in robustness if r["checkpoint_updates"] == 8000]
        assert sorted(r["evaluation_seed"] for r in final_robust) == [1, 2, 3, 4, 5]
        runtime = read(ROOT / f"reports/microbatch/{name}/result.json")
        expected_new = 6000 if name == NAMES[0] else 8000
        assert runtime["status"] == "passed" and runtime["additional_updates"] == expected_new
        update_rows = [json.loads(line) for line in (ROOT / f"reports/microbatch/{name}/updates.jsonl").read_text().splitlines()]
        finite = all(np.isfinite(v) for r in update_rows for key in ("metrics", "world_model_metrics") for v in r[key].values())
        assert finite
        trajectory = np.load(folder / "final.trajectory.npz")
        motion = np.load(ROOT / final["motion"])
        joint_rmse = np.sqrt(np.mean((trajectory["qpos"][:, 7:] - trajectory["reference_qpos"][:, 7:]) ** 2, axis=0))
        joint_errors = {str(key): float(value) for key, value in zip(motion["joint_names"], joint_rmse)}
        summary = dict(name=name, label=label, final=final, runtime=runtime,
                       final_joint_rmse_rad=joint_errors,
                       completed_reset_episodes=sum(r["completed_clip"] for r in final_robust), reset_episodes=5,
                       reset_mean_full_clip_return=float(np.mean([r["normalized_full_clip_return"] for r in final_robust])),
                       reset_std_full_clip_return=float(np.std([r["normalized_full_clip_return"] for r in final_robust])),
                       reset_mean_root_error_m=float(np.mean([r["mean_root_position_error_m"] for r in final_robust])),
                       reset_mean_yaw_error_deg=float(np.mean([r["mean_abs_yaw_error_deg"] for r in final_robust])),
                       all_update_metrics_finite=finite, evaluations=[unique[k] for k in sorted(unique)],
                       robustness=robustness)
        records.append(summary)

    for offset, kind in [(0, "short"), (2, "long")]:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        panels = [("normalized_full_clip_return", "Full-clip normalized return"),
                  ("completion_fraction", "Completion fraction"),
                  ("joint_position_rmse_rad", "Joint RMSE (rad)"),
                  ("mean_root_position_error_m", "Mean root error (m)"),
                  ("mean_abs_yaw_error_deg", "Mean absolute yaw error (deg)"),
                  ("body_orientation_fraction", "Body orientation reward / maximum")]
        for rec, label in zip(records[offset:offset+2], ["No anchor", "Anchor"]):
            rows = rec["evaluations"]
            for ax, (field, title) in zip(axes.flat, panels):
                vals = [(r.get("reward_fraction_of_term_max", {}).get("motion_body_ori") if field == "body_orientation_fraction" else r.get(field)) for r in rows]
                ax.plot([r["checkpoint_updates"] for r in rows], [np.nan if v is None else v for v in vals], "o-", label=label)
                ax.set(title=title, xlabel="Policy updates")
                ax.grid(alpha=.25)
                ax.legend()
        fig.suptitle(f"T800 {kind} motion: one training seed; deterministic evaluation")
        fig.savefig(OUT / f"{kind}-learning.png", dpi=150)
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        motion = np.load(ROOT / records[offset]["final"]["motion"])
        root_index = list(motion["body_names"]).index("LINK_BASE")
        ref_pos = motion["body_pos_w"][:, root_index]
        axes[0].plot(ref_pos[:, 0], ref_pos[:, 1], "k--", label="Reference")
        for rec, label in zip(records[offset:offset+2], ["No anchor", "Anchor"]):
            tr = np.load(ROOT / f"reports/training/{rec['name']}/final.trajectory.npz")
            axes[0].plot(tr["qpos"][:, 0], tr["qpos"][:, 1], label=label)
            yaw = tr["yaw_error_deg"].copy()
            yaw[1:][np.abs(np.diff(yaw)) > 180] = np.nan
            axes[1].plot(tr["time_s"], yaw, label=label)
        axes[0].set(title="Final checkpoint: world trajectory", xlabel="x (m)", ylabel="y (m)", aspect="equal")
        axes[1].set(title="Heading error (no alignment)", xlabel="Time (s)", ylabel="Yaw error (deg)")
        for ax in axes:
            ax.grid(alpha=.25)
            ax.legend()
        fig.savefig(OUT / f"{kind}-trajectory.png", dpi=150)
        plt.close(fig)

    validation = read(OUT / "validation/step1.json")
    long_audit = read(OUT / "long-motion-audit.json")
    deltas = {}
    baseline_2000 = read(ROOT / f"reports/training/{NAMES[0]}/baseline.json")
    comparison_fields = ("normalized_full_clip_return", "mean_root_position_error_m", "mean_abs_yaw_error_deg", "joint_position_rmse_rad")
    deltas["short_noanchor_2000_to_8000"] = {
        key: records[0]["final"][key] - baseline_2000[key] for key in comparison_fields
    }
    for offset, kind in [(0, "short"), (2, "long")]:
        if offset + 1 >= len(records):
            deltas[kind] = None
            continue
        a, b = records[offset]["final"], records[offset+1]["final"]
        deltas[kind] = {key: b[key] - a[key] for key in comparison_fields}
    diagnosis = {name: read(OUT / "local-diagnosis" / name) for name in
                 ("diagnosis.json", "wm-action-sensitivity.json", "action-parameterization.json")}
    alignment = read(OUT / "alignment-targets.json")
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(), status="delivered_revised_scope" if cancelled else "complete",
                   new_formal_updates=sum(r["runtime"]["additional_updates"] for r in records),
                   reused_updates=2000, anchor_smoke_updates=4, cancelled_stage=cancelled,
                   cancelled_stage_last_recorded_updates=study_state.get("cancelled_stage_last_recorded_updates"),
                   alignment_targets=alignment,
                   comparisons=deltas, baseline_2000=baseline_2000, runs=records, validation=validation,
                   local_tracking_diagnosis=diagnosis, long_motion_audit=long_audit)
    (OUT / "delivery.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    nominal_rows = []
    robust_rows = []
    terms_rows = []
    wm_rows = []
    joint_rows = []
    for r in records:
        f = r["final"]
        nominal_rows.append([r["label"], f"{f['steps']}/{f['reference_frames']}", "是" if f["completed_clip"] else "否",
                             number(f["normalized_full_clip_return"]), number(f["normalized_return"]),
                             number(f["mean_root_position_error_m"]), number(f["mean_abs_yaw_error_deg"], 2), number(f["joint_position_rmse_rad"])])
        robust_rows.append([r["label"], f"{r['completed_reset_episodes']}/5", number(r["reset_mean_full_clip_return"]),
                            number(r["reset_std_full_clip_return"]), number(r["runtime"]["elapsed_seconds"]/3600, 2),
                            number(r["runtime"]["torch_peak_allocated_gib"], 2)])
        frac = f["reward_fraction_of_term_max"]
        terms_rows.append([r["label"]] + [number(100*frac[k], 1)+"%" for k in (
            "motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel", "motion_body_ang_vel")])
        on_policy = f.get("world_model_eval_nse_by_horizon") or []
        fixed = f.get("world_model_fixed_test_nse_by_horizon") or []
        wm_rows.append([r["label"]] + [number(values[h-1], 5) if len(values) >= h else "无足够有效窗口"
                                       for values in (on_policy, fixed) for h in (1, 8, 32)])
        largest = sorted(r["final_joint_rmse_rad"].items(), key=lambda item: item[1], reverse=True)[:5]
        joint_rows.append([r["label"], "; ".join(f"{key}: {value:.3f}" for key, value in largest),
                           number(f["torque_saturation_fraction"] * 100, 3) + "%",
                           number(f["hard_joint_limit_fraction"] * 100, 3) + "%"])
    noanchor_8000 = records[0]["final"]
    conclusions = [
        f"延长训练（短动作、无 anchor）：2000 → 8000 次更新，完整片段回报 {baseline_2000['normalized_full_clip_return']:.3f} → {noanchor_8000['normalized_full_clip_return']:.3f}，"
        f"根位置误差 {baseline_2000['mean_root_position_error_m']:.3f} → {noanchor_8000['mean_root_position_error_m']:.3f} m，"
        f"平均绝对 yaw 误差 {baseline_2000['mean_abs_yaw_error_deg']:.2f}° → {noanchor_8000['mean_abs_yaw_error_deg']:.2f}°，"
        f"关节 RMSE {baseline_2000['joint_position_rmse_rad']:.3f} → {noanchor_8000['joint_position_rmse_rad']:.3f} rad。"
    ]
    for i, label in [(0, "短动作"), (2, "60 秒连续动作")]:
        if i + 1 >= len(records):
            f = records[i]["final"]
            conclusions.append(f"{label}：无 anchor 最终回放完整片段回报 {f['normalized_full_clip_return']:.3f}，根位置平均误差 {f['mean_root_position_error_m']:.3f} m，平均绝对 yaw 误差 {f['mean_abs_yaw_error_deg']:.2f}°，关节 RMSE {f['joint_position_rmse_rad']:.3f} rad。长动作 anchor 已取消，不能给出其收益结论。")
            milestone = next(r for r in read(ROOT / f"reports/training/{records[i]['name']}/evaluations.json") if r["checkpoint_updates"] == 8000)
            conclusions.append(f"同一长动作 8000 次检查点的两次无重置噪声回放分别得到根位置误差 {milestone['mean_root_position_error_m']:.3f} / {f['mean_root_position_error_m']:.3f} m、回报 {milestone['normalized_return']:.3f} / {f['normalized_return']:.3f}；报告主表固定使用 final.json，保留长时程回放的数值敏感性。")
            continue
        a, b = records[i], records[i+1]
        delta = b["final"]["normalized_full_clip_return"] - a["final"]["normalized_full_clip_return"]
        conclusions.append(f"{label}：anchor 相对无 anchor 的完整片段归一化回报变化为 {delta:+.3f}；重置扰动完成次数为 {b['completed_reset_episodes']}/5，对照为 {a['completed_reset_episodes']}/5。")
        conclusions.append(
            f"{label}同预算误差（无 anchor → anchor）：根位置 {a['final']['mean_root_position_error_m']:.3f} → {b['final']['mean_root_position_error_m']:.3f} m；"
            f"平均绝对 yaw {a['final']['mean_abs_yaw_error_deg']:.2f}° → {b['final']['mean_abs_yaw_error_deg']:.2f}°；"
            f"关节 RMSE {a['final']['joint_position_rmse_rad']:.3f} → {b['final']['joint_position_rmse_rad']:.3f} rad。"
        )
        if not a["final"]["completed_clip"] or not b["final"]["completed_clip"]:
            conclusions.append(f"{label}至少一组未完成整段，误差均值的统计时长不同；以完整片段回报、完成度和失败时间共同判断，不能仅比较存活期间的误差均值。")
    links = ''.join(f'<li>{html.escape(r["label"])}：<a href="../training/{r["name"]}/final.json">最终评估</a> · <a href="../training/{r["name"]}/final.png">详细曲线</a> · <a href="../training/{r["name"]}/robustness.json">扰动评估</a> · <a href="../../configs/{r["name"]}.json">冻结协议</a> · <a href="../../logs/{r["name"]}/ckpt_latest.pt">8000 次检查点</a></li>' for r in records)
    doc = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>T800 上游指标对齐与误差诊断报告</title><style>
body{{font-family:system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#17212b;line-height:1.75;background:#fafbfc}}
h1,h2{{line-height:1.3}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{border:1px solid #dbe1e8;padding:9px;text-align:left}}th{{background:#edf2f7}}img{{width:100%;background:white;margin:12px 0}}a{{color:#1756a9}}.note{{background:#eef4fb;border-left:4px solid #416c9a;padding:14px 18px}}code{{background:#eef0f3;padding:2px 5px}}
</style><body><h1>T800 实验、上游指标对齐与误差诊断报告</h1><p>生成时间：{summary['generated_utc']}。{scope_text}</p>
<h2>下一阶段要对齐的指标</h2>{table(['指标','上游公开参照','对齐要求'], [[r['metric'], r['reference'], r['comparison_rule']] for r in alignment['metrics']])}
<p>先建立 G1 同机器人、同动作、同训练预算的复现控制组，再将同一评估口径迁移到 T800。e1c 的训练回报与整段回报是不同协议；e2 的局部奖励来自 40 条 LAFAN1，不能混成同一组验收结果。公开数值作为参考线，不捏造重复实验置信区间。完整计划和原因排序见<a href="alignment-targets.json">指标与验证计划</a>。</p>
<h2>实测结论</h2>{''.join('<p>'+html.escape(c)+'</p>' for c in conclusions)}
<p class="note">这是单个训练种子的实验结果。五组重置扰动是评估条件的重复，不是五次独立训练；不把本结果等同于上游 40 条 LAFAN1 的 0.904 基准。主比较固定使用 8000 次最终检查点，不按回报挑选中途最优检查点；完整学习曲线另列。</p>
<h2>关节 RMSE 是否达到上游水平</h2>
<p>固定上游版本的公开实验记录与 benchmark 没有提供本报告定义的关节角 RMSE，因此上游对应的 rad 数值未知，不能把 T800 的约 0.3 rad 当作已经复现的正常水平。算法流程对齐不保证不同机器人上的跟踪质量相同。</p>
<p>上游无 anchor 的 LAFAN1 e2 实验公布了局部奖励：相对刚体位置约 95%、相对刚体姿态约 75%、线速度约 50%、角速度约 55%（均为该项上限的比例）。本报告最终奖励表可与其作诊断性参照；机器人、动作集合和训练阶段不同，不能视为严格配对比较，也不能由奖励反推出关节 RMSE。</p>
<p>预算口径也有区别：上游单动作 e1c 为 8104 次策略更新；上游 anchor 实验的 0.904 出现在 48000 次环境迭代，按 UTD=0.4 约为 19200 次策略更新，并不是 8000 次。因此本实验验证的是用户指定的 8000 次预算，不声称复现了上游 0.904 所用的全部训练预算。参见<a href="https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/docs/EXPERIMENTS.md">固定版本实验记录</a>。</p>
<h2>最终检查点：每组累计 8000 次策略更新</h2>{table(['实验','实际步数 / 参考步数','整段完成','完整片段回报','存活期间回报','根位置误差 m','平均 |yaw| °','关节 RMSE rad'], nominal_rows)}
<p>完整片段回报 = 已取得奖励之和 / (0.1 × 全部参考帧数)，未播放部分记零。存活期间回报保留旧评估定义。误差均值仅覆盖实际播放的状态。</p>
<h2>五组重置扰动与资源记录</h2>{table(['实验','完成次数','完整回报均值','标准差','本次训练小时','Torch 峰值 GiB'], robust_rows)}
<p>短动作无 anchor 的本次训练时间仅统计新增 6000 次；前 2000 次来自原实验。训练噪声和域随机化保持上游设置；评估关闭动作/观测噪声和域随机化，扰动评估仅启用原始重置扰动。</p>
<h2>最终奖励分解</h2>{table(['实验','根位置','根姿态','相对刚体位置','相对刚体姿态','线速度','角速度'], terms_rows)}
<p>表中百分比为每项奖励相对于该项最大值的比例。相对刚体位置/姿态已按机器人 anchor 的 XY 和 yaw 对齐；速度奖励仍在世界坐标系计算。Yaw 对齐速度诊断保存在各评估 JSON 中，它是固定轨迹的离线诊断，不能视为重新训练 anchor 的收益上限。</p>
<h2>局部关节误差定位</h2>{table(['实验','RMSE 最大的五个关节（rad）','力矩饱和采样比例','硬限位采样比例'], joint_rows)}
<p>逐关节误差由同一时刻保存的机器人与参考 qpos 独立计算，完整数值见 delivery.json。力矩与限位比例仅采样每个控制步的最后一个物理子步，低比例不能排除子步内饱和或其他控制限制。这些诊断定位残余误差，不单独证明误差的原因。</p>
<h2>残余误差排查与下一步</h2>
<p>算法计算逻辑的对齐不等于动作参数化的规则完全一致：上游 G1 用 <code>action_scale = 0.25 × effort_limit / stiffness</code>，本适配使用 T800 RL Lab 固定尺度。按上游规则，T800 肩部 yaw / 肘部 pitch 分别应为 0.8 / 0.8，当前为 0.05 / 0.2；在保持原始动作变化惩罚系数时，这会改变物理目标角变化对应的惩罚、探索幅度和优化尺度。</p>
<p>将短动作首帧参考角度直接换成 PD 目标，并假设上一原始动作为零，当前尺度下动作变化惩罚为 −0.572；采用上游尺度规则时为 −0.016，而正奖励单步总上限为 0.1。这是代数诊断，不是重新训练收益，也不代表必须一步切换到参考目标。稳定跟随参考目标时（排除首步）的该惩罚均值仅约 −0.000693；所以不能用动作变化惩罚单独解释全部恒定姿态偏差。<a href="local-diagnosis/action-parameterization.json">计算记录</a>。</p>
<p>固定直立躯干、保留重力、关闭接触的 CPU PD 隔离试验中，左右肘跟踪参考目标的 RMSE 约 0.05 rad，明显小于训练策略的误差。该试验支持关节伺服具备更好跟踪能力，但不证明整段自由行走的动力学可行性。<a href="local-diagnosis/diagnosis.json">完整隔离试验</a>。</p>
<p>另外，在每个短动作最终检查点保存的 32 个策略状态上，CPU 单步重放与记录的 GPU 关节位置最大逐关节 RMSE 小于 4×10⁻⁷ rad；世界模型肘部单步角度预测误差约 0.0003–0.0006 rad。肘部 pitch 的同关节动作导数与 CPU 物理有限差分相对 L2 误差约 0.7–0.8%，符号一致。两种扰动步长均检查。没有发现该局部单步导数方向颠倒；仍未验证全部交叉导数、长期闭环梯度或离策略状态。<a href="local-diagnosis/wm-action-sensitivity.json">世界模型动作敏感度诊断</a>。</p>
<p><strong>建议优先做单变量短动作实验：只将 action scale 改为上游的力矩/Kp 规则，保留默认姿态、PD、奖励、数据和预算，比较关节 RMSE 与局部奖励。</strong>这项实验尚未运行，已完成对照的参数未因此改变。当前证据将动作参数化列为优先验证的原因，尚不能宣布其为唯一根因；尺度还同时改变物理动作噪声和惩罚含义，后续应分开验证。膝部接触动力学、长期模型梯度以及软件栈/重计算的长期数值影响仍有未验证部分。</p>
<h2>世界模型预测误差</h2>{table(['实验','当前策略 H=1','当前策略 H=8','当前策略 H=32','固定窗口 H=1','固定窗口 H=8','固定窗口 H=32'], wm_rows)}
<p>数值为按上游状态尺度归一化的均方误差，没有米或弧度单位。当前策略测试取自各自新回放，所覆盖的状态分布不同；固定窗口在各组间相同，但来自最初短动作的未训练策略轨迹。二者一起用于判断模型学习和分布变化。</p>
<h2>短动作学习与轨迹</h2><img src="short-learning.png"><img src="short-trajectory.png">
<h2>连续 60 秒动作学习与轨迹</h2><img src="long-learning.png"><img src="long-trajectory.png">
<h2>实验与实现</h2><ul>
<li>同一上游算法快照 ba610534；核心训练源码及 optimizations 未改变。新增工作位于训练入口、阶段驱动、评估与报告脚本。</li>
<li>4096 环境、策略 batch 4096、H=32、UTD=0.4；原有网络宽度、critic、奖励、随机化和 320k 环境步学习率规划保持一致。IEEE FP32 矩阵乘法、仅策略激活重计算。</li>
<li>Anchor 开启时目标输入由 600 维变为 609 维，物理环境与想象轨迹均使用上游已有的 9 维相对位姿和噪声设定；策略和 critic 输入同步变化。</li>
<li>短动作无 anchor 从原 2000 次检查点恢复，新增 6000 次；anchor 从随机初始化训练 8000 次。长动作无 anchor 从随机初始化训练 8000 次；长动作 anchor 的执行状态见报告开头。</li>
<li>长动作由原始 walk2_subject1.bvh 的 [300:2101] 帧连续重定向得到，30 Hz 输入、50 Hz 输出，时长 60 秒，未循环拼接短动作；数据审计通过。</li>
<li>每 2000 次更新保存不可变检查点，各阶段训练结束后独占 GPU 做确定性回放和相同种子的重置扰动评估。原始检查点和日志均保留。</li>
</ul>
<p>算法来源：<a href="https://github.com/CuboiLeo/SuperMimic-Reproduced/tree/ba6105349069bf040f1fdad60873d13138d505a2">固定上游版本</a>；<a href="https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/docs/EXPERIMENTS.md">上游实验记录</a>。本次测试针对 T800 和指定单条动作，采用独立的数据范围和硬件。</p>
<h2>验证与解释边界</h2><p>新增指标在保存的真实轨迹和精确参考帧上独立复算通过，奖励各项之和与环境返回值一致；anchor 完整规模 4 次更新检查通过，609 维输入和精确检查点保存通过。<a href="validation/step1.json">完整验证记录</a>。</p>
<p>旧检查点的严格 GPU 回放 1e-5 容差未在所有重复中满足：根位置均值差约 {validation['legacy_gpu_rollout_abs_difference']['mean_root_position_error_m']:.6f} m，归一化回报差约 {validation['legacy_gpu_rollout_abs_difference']['normalized_return']:.6f}。记录完整保留，不宣称逐位等价或把差异归因于某个尚未证实的原因。</p>
<p>上游终止标签错位行为仍保留，没有与 anchor 改动混合。不同 PyTorch/CUDA 软件栈及训练后数值对照的既有局限仍成立。世界模型固定测试窗口沿用短动作原始未训练策略的数据；长动作新轨迹的模型误差另行报告。单次短期模型预测准确不证明长期闭环梯度准确。</p>
<p>长动作参考数据检查：<a href="long-motion-audit.json">FK、关节限位与足部几何审计</a>。三组已完成训练的更新指标均有限；这不等于策略必然收敛或跟踪误差达标。</p>
<h2>交付文件</h2><ul>{links}<li><a href="delivery.json">完整机器可读结果及每个检查点评估</a></li><li><a href="status.json">执行状态</a></li></ul>
</body></html>'''
    (OUT / "report.html").write_text(doc)
    print(json.dumps({"report": str(OUT / 'report.html'), "comparisons": deltas}, indent=2))


if __name__ == "__main__":
    main()
