"""Progress and evidence for the G1-u4052 handoff and corrected T800 baseline."""
from datetime import datetime,timezone
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/t800_fkfix_study'
RUN='t800_walk1_full_fkfixed_8104'
G1='g1_walk1_e1c_fkfixed_8104'


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    OUT.mkdir(exist_ok=True)
    folder=ROOT/'reports/training'/RUN
    stage=read(folder/'status.json');train=read(ROOT/'reports/microbatch'/RUN/'result.json')
    handoff=read(OUT/'handoff-status.json');g1folder=ROOT/'reports/training'/G1
    state=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status=(stage or handoff or {}).get('status','preparing'),
        handoff=handoff,training=train,stage=stage,protocol=read(OUT/'protocol.json'),
        profile_audit=read(OUT/'profile-audit.json'),
        full_batch_probe=read(ROOT/'reports/microbatch/t800_fkfixed_fullbatch_policy_probe/result.json'),
        joint_error_breakdown=read(OUT/'joint-error-breakdown.json'),
        g1_4052=read(g1folder/'eval_ckpt_u0004052.json'),g1_4052_upstream=read(g1folder/'eval_ckpt_u0004052.upstream.json'),
        g1_old_4052=read(ROOT/'reports/training/g1_walk1_e1c_aligned_8104/eval_ckpt_u0004052.json'),
        g1_old_4052_upstream=read(OUT/'g1-old-4052.upstream.json'),
        evaluations={str(u):read(folder/f'eval_ckpt_u{u:07d}.json') for u in [2000,4052,8104]},
        benchmarks={str(u):read(folder/f'eval_ckpt_u{u:07d}.upstream.json') for u in [2000,4052,8104]},
        final=read(folder/'final.json'),robustness=read(folder/'robustness.json'))
    log=folder/'training.log'
    if log.exists():
        lines=[l for l in log.read_text(errors='replace').splitlines() if l.startswith('it ')]
        state['latest_training_log']=lines[-1] if lines else None
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(state,indent=2,ensure_ascii=False)+'\n');tmp.replace(OUT/'status.json')
    esc=html.escape
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T800 FK 修复训练</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#243344}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd7df;padding:9px;text-align:left}th{background:#edf2f7}.note{padding:16px;background:#edf5f1}code{overflow-wrap:anywhere}a{color:#145f95}</style><body><h1>T800 修复后完整 walk1 基线</h1>'''
    doc+=f'<p>状态：{esc(state["status"])}；更新 UTC：{esc(state["updated_utc"])}</p>'
    doc+='<p class="note">按用户指令：G1 在 4052 次保存并完成真实跟踪与上游 benchmark 评估后暂停，再切换 T800。新 T800 从 seed=0 开始，完整 walk1 13066 帧、50 Hz；4096 环境、4096 batch、32 步想象、8104 次更新。保留原学习率计划、policy 重计算和 IEEE FP32。</p>'
    doc+='<p>T800 使用已经核对的物理参数，25 个动作关节；14 个身体跟踪点、LINK_WAIST_YAW anchor、LINK_BASE 根状态；动作尺度为 0.25×T800 力矩上限/Kp；不开额外 9 维 anchor 观测。训练、评估统一安装上述配置和编译 FK 修复，检查点记录并核对四组版本元数据。</p>'
    doc+='<h2>G1 同预算结果（4052 次）</h2><table><tr><th>组别</th><th>上游 benchmark return</th><th>诊断 evaluation return</th><th>关节 RMSE / rad</th><th>根位置均误差 / m</th><th>yaw 均误差 / °</th><th>诊断回放帧数 / 总帧数</th></tr>'
    for label,r,b in [('旧组',state['g1_old_4052'],state['g1_old_4052_upstream']),('修复组',state['g1_4052'],state['g1_4052_upstream'])]:
        if r:
            br=f'{b["clips"][0]["norm_return"]:.4f}' if b else '等待评估'
            doc+=f'<tr><td>{label}</td><td>{br}</td><td>{r["normalized_return"]:.4f}</td><td>{r["joint_position_rmse_rad"]:.4f}</td><td>{r["mean_root_position_error_m"]:.3f}</td><td>{r["mean_abs_yaw_error_deg"]:.2f}</td><td>{r["steps"]} / {r["reference_frames"]}</td></tr>'
        else:doc+=f'<tr><td>{label}</td><td colspan="6">等待检查点及评估</td></tr>'
    doc+='</table><p>G1 暂停保留了模型和优化器检查点。上游恢复流程会重新填充 replay、不会恢复全部 RNG，因此以后续训不宣称与未中断训练逐步等价。</p>'
    g1=state['g1_4052']
    if g1 and not g1['completed_clip']:
        doc+=f'<p class="note"><b>G1 评估口径说明：</b>修复组 IEEE FP32 诊断回放在 {g1["steps"]/50:.2f} 秒因 {esc(str(g1["termination_reasons"]))} 提前终止，其 {g1["normalized_return"]:.4f} 是已执行步平均回报，计入未执行帧后的完整动作回报为 {g1["normalized_full_clip_return"]:.4f}。未经修改的上游 benchmark 使用默认 TF32，此次完整完成动作；同预算 return 比较使用表中两组上游 benchmark。两种执行设置不能混为一次完全相同的回放。</p>'
    audit=state['profile_audit']
    if audit:
        doc+=f'<p>T800 联合检查：{audit["status"]}；真实环境与解析奖励最大误差 {max(audit["unscaled_reward_max_abs_error"].values()):.2e}；编译/非编译单步奖励差 {audit["compiled_imagstep_outputs"]["reward"]["max_abs"]:.2e}，动作梯度方向余弦 {audit["compiled_reward_gradients"]["action"]["cosine"]:.9f}。</p>'
    probe=state['full_batch_probe']
    if probe:
        peak=probe.get('torch_peak_allocated_gib')
        doc+=f'<p>完整 batch 验证：{probe["status"]}；PyTorch 峰值：{peak if peak is not None else "等待结果"} GiB。</p>'
    doc+=f'<h2>T800 训练和评估</h2><p>已更新 {(train or {}).get("updates",0)} / 8104。<br><code>{esc(state.get("latest_training_log") or "尚无训练回报日志")}</code></p>'
    doc+='<table><tr><th>更新</th><th>Evaluation return</th><th>完整动作归一化回报</th><th>关节 RMSE / rad</th><th>根位置均误差 / m</th><th>完成全部帧</th></tr>'
    for u,r in state['evaluations'].items():
        if r:doc+=f'<tr><td>{u}</td><td>{r["normalized_return"]:.4f}</td><td>{r["normalized_full_clip_return"]:.4f}</td><td>{r["joint_position_rmse_rad"]:.4f}</td><td>{r["mean_root_position_error_m"]:.3f}</td><td>{r["completed_clip"]}</td></tr>'
        else:doc+=f'<tr><td>{u}</td><td colspan="5">等待训练与评估</td></tr>'
    doc+='</table><p>该组同时采用新完整 walk1 参考和已对齐的 T800 配置，不能与历史短 walk2 实验解释为只修改 FK 的单变量对照。训练回报、实际执行步平均回报和将未执行帧计零的完整动作回报分别记录；关节误差另记录动作前和奖励时刻两种口径。</p>'
    breakdown=state['joint_error_breakdown']
    if breakdown:
        doc+='<h2>关节误差分解</h2><table><tr><th>更新</th><th>全部 25 关节 RMSE / rad</th><th>非头部 23 关节 RMSE / rad</th><th>头部 2 关节 RMSE / rad</th><th>头部占总关节平方误差</th></tr>'
        for u,r in breakdown['checkpoints'].items():
            doc+=f'<tr><td>{u}</td><td>{r["overall_rmse_rad"]:.4f}</td><td>{r["non_head_rmse_rad"]:.4f}</td><td>{r["head_rmse_rad"]:.4f}</td><td>{100*r["head_fraction_of_total_joint_squared_error"]:.1f}%</td></tr>'
        doc+='</table><p>这些数值从已保存的完整轨迹计算，没有重跑仿真或改动策略。头部没有进入 14 刚体跟踪奖励，最终占总关节平方误差约 93%；全部关节 RMSE 的后期回升主要来自头部，非头部关节仍明显改善。这也暴露了目前头部受控但缺少直接跟踪目标的适配限制。</p>'
    if state['status']=='finished':
        doc+='<h2>交付结论</h2><p>G1 已按要求在 4052 次暂停；T800 完成 8104 次训练、三组检查点评估和五组重置扰动，阶段没有评估错误。T800 最终 training return 约 0.77，benchmark return 约 0.635；非头部关节 RMSE 约 0.0665 rad，五组扰动全部完成 261 秒动作。</p><p><b>仍未解决：</b>最终根位置平均误差约 9.61 m、yaw 平均误差约 83.25°；不能把未摔倒和局部姿态准确当作全局轨迹跟踪成功。下一步应明确头部控制目标，并在相同头部配置下做额外 9 维 anchor 观测对照。当前没有启动新的训练。</p>'
    links=['protocol.json','status.json','handoff-status.json','profile-audit.json','probe-evaluation.json','probe-benchmark.json','g1-old-4052.upstream.json','joint-error-breakdown.json']
    doc+='<p>'+' · '.join(f'<a href="{p}">{p}</a>' for p in links if (OUT/p).exists())+'</p></body></html>'
    (OUT/'report.html').write_text(doc)
    print(json.dumps(dict(status=state['status'],updates=(train or {}).get('updates',0))))


if __name__=='__main__':main()
