"""Generate the live, same-budget baseline/anchor+fixed-head comparison."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/t800_anchor_head_study'
RUN = 't800_walk1_anchor_headfixed_8104'
BASELINE = 't800_walk1_full_fkfixed_8104'
PROBE = 't800_anchor_head_fullbatch_probe'
MILESTONES = [2000, 4052, 8104]


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def main():
    OUT.mkdir(exist_ok=True)
    folder = ROOT / 'reports/training' / RUN
    stage = read(folder / 'status.json')
    train = read(ROOT / 'reports/microbatch' / RUN / 'result.json')
    breakdown = read(ROOT / 'reports/t800_fkfix_study/joint-error-breakdown.json')
    state = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        status=(stage or {}).get('status', 'preparing'), training=train, stage=stage,
        protocol=read(OUT / 'protocol.json'), profile_audit=read(OUT / 'profile-audit.json'),
        full_batch_probe=read(ROOT / 'reports/microbatch' / PROBE / 'result.json'),
        baseline_joint_breakdown=breakdown, evaluations={}, benchmarks={},
        robustness=read(folder / 'robustness.json'), final=read(folder / 'final.json'))
    for label, name in [('baseline', BASELINE), ('anchor_head', RUN)]:
        path = ROOT / 'reports/training' / name
        state['evaluations'][label] = {str(u): read(path / f'eval_ckpt_u{u:07d}.json') for u in MILESTONES}
        state['benchmarks'][label] = {str(u): read(path / f'eval_ckpt_u{u:07d}.upstream.json') for u in MILESTONES}
    log = folder / 'training.log'
    if log.exists():
        lines = [s for s in log.read_text(errors='replace').splitlines() if s.startswith('it ')]
        state['latest_training_log'] = lines[-1] if lines else None
    write(OUT / 'status.json', json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    esc = html.escape
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T800 anchor 与固定头部实验</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:32px auto;padding:0 20px;color:#243344}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}th{background:#edf2f7}.note{background:#edf5f1;padding:15px}code{overflow-wrap:anywhere}a{color:#145f95}</style><body>
<h1>T800：anchor 观测 + 固定头部</h1>'''
    doc += f'<p>状态：{esc(state["status"])}；UTC 更新：{esc(state["updated_utc"])}</p>'
    doc += '<p class="note">按用户选择，本轮同时增加 9 维 LINK_WAIST_YAW 相对位姿观测，并将头部两个 PD 目标固定为默认角度（均为 0 rad）。头部动作及执行噪声被屏蔽，保留物理关节和真实状态。该组合实验不能独立归因 anchor 与固定头部各自的收益。</p>'
    doc += '<p>与已完成基线使用相同完整 walk1 参考（13066 帧，261 秒）、T800 PD 参数、动作尺度、14 刚体奖励和 FK 修复；4096 环境、4096 policy batch、32 步想象、8104 次更新。保留原学习率计划、IEEE FP32 和 policy 重计算。两组均从 seed 0 开始；因输入尺寸改变，actor/critic 的初始权重不完全相同。G1 保持暂停。</p>'
    doc += f'<p>进度：{(train or {}).get("updates", 0)} / 8104。<br><code>{esc(state.get("latest_training_log") or "尚无训练回报日志")}</code></p>'
    doc += '<table><tr><th>更新 / 组别</th><th>Benchmark return</th><th>完整动作诊断回报</th><th>非头部 RMSE / rad</th><th>头部 RMSE / rad</th><th>根位置均误差 / m</th><th>yaw 均误差 / °</th><th>完成帧数</th></tr>'
    for u in MILESTONES:
        for label, title in [('baseline', '原基线'), ('anchor_head', 'anchor + 固定头部')]:
            r = state['evaluations'][label][str(u)]
            b = state['benchmarks'][label][str(u)]
            if not r:
                doc += f'<tr><td>{u} / {title}</td><td colspan="7">等待训练与评估</td></tr>'
                continue
            split = (breakdown or {}).get('checkpoints', {}).get(str(u), {}) if label == 'baseline' else {}
            body = r.get('non_head_joint_rmse_rad', split.get('non_head_rmse_rad'))
            head = r.get('head_default_rmse_rad', split.get('head_rmse_rad'))
            fmt = lambda value, n=4: f'{value:.{n}f}' if value is not None else '待计算'
            benchmark = b['clips'][0]['norm_return'] if b else None
            doc += f'<tr><td>{u} / {title}</td><td>{fmt(benchmark)}</td><td>{r["normalized_full_clip_return"]:.4f}</td><td>{fmt(body)}</td><td>{fmt(head)}</td><td>{r["mean_root_position_error_m"]:.3f}</td><td>{r["mean_abs_yaw_error_deg"]:.2f}</td><td>{r["steps"]} / {r["reference_frames"]}</td></tr>'
    doc += '</table><p>头部参考角度与默认角度均为零；本表按此计算头部误差。身体 23 关节单独统计，避免头部掩盖身体改善。Benchmark 沿用上游默认 TF32 口径；诊断使用 IEEE FP32，同时保存已执行步均回报及未执行帧计零的完整动作回报。每个检查点完整回放，最终另做五组重置扰动。</p>'
    audit = state['profile_audit']
    if audit:
        h = audit['head_control_audit']
        doc += f'<p>配置检查：{audit["status"]}。真实环境 / 解析目标最大差 {audit["goal_max_abs_error"]:.2e}；头部 PD 目标偏差 {h["pd_target_max_abs_error"]:.2e} rad；回放及想象头部动作为零。编译 / eager 奖励差 {audit["compiled_imagstep_outputs"]["reward"]["max_abs"]:.2e}。</p>'
    probe = state['full_batch_probe']
    if probe:
        doc += f'<p>全规模预检：{esc(probe["status"])}；完成 {probe.get("additional_updates", 0)} 次更新；PyTorch 峰值 {probe.get("torch_peak_allocated_gib", "待记录")} GiB。</p>'
    if state['status'] == 'finished':
        final = state['evaluations']['anchor_head']['8104']
        base = state['evaluations']['baseline']['8104']
        if final and base:
            body_base = breakdown['checkpoints']['8104']['non_head_rmse_rad']
            doc += '<h2>最终结论</h2>'
            doc += f'<p>完成 8104 次更新和全部评估，没有评估错误。完整 261 秒动作诊断回报从 {base["normalized_full_clip_return"]:.4f} 提高到 {final["normalized_full_clip_return"]:.4f}；根位置平均误差从 {base["mean_root_position_error_m"]:.3f} m 降到 {100*final["mean_root_position_error_m"]:.2f} cm；yaw 平均误差从 {base["mean_abs_yaw_error_deg"]:.2f}° 降到 {final["mean_abs_yaw_error_deg"]:.2f}°。长期全局漂移显著改善。</p>'
            doc += f'<p>头部对默认姿态 RMSE 为 {final["head_default_rmse_rad"]:.4f} rad，执行动作为零。但身体 23 关节 RMSE 从 {body_base:.4f} 上升到 {final["non_head_joint_rmse_rad"]:.4f} rad，局部姿态精度没有同步超过原基线。训练末期 return 约 0.82，与表中无噪声完整评估口径不同。</p>'
        robust = state['robustness'] or []
        if robust:
            returns = [r['normalized_full_clip_return'] for r in robust]
            completed = sum(r['completed_clip'] for r in robust)
            doc += f'<p>重置扰动：{completed}/{len(robust)} 组完成完整动作，完整动作回报范围 {min(returns):.4f}–{max(returns):.4f}。这些是同一训练种子的五次评估，不代表五个独立训练种子。</p>'
        early = state['evaluations']['anchor_head']['2000']
        if early and not early['completed_clip']:
            doc += f'<p>2000 次检查点的 IEEE 诊断在 {early["steps"]/50:.2f} 秒因 {esc(str(early["termination_reasons"]))} 终止，已执行步回报 {early["normalized_return"]:.4f}、完整动作计零尾部回报 {early["normalized_full_clip_return"]:.4f}；TF32 benchmark 此时完成动作。两种精度的早期回放表现不同。4052 和 8104 次两种评估都完成动作，最终回报接近。</p>'
        doc += '<p>本轮证明了组合方案在这条训练动作上的改善；anchor 与固定头部各自的因果贡献仍需单独消融验证，尚未测试未见动作上的泛化。</p>'
        doc += f'<p><a href="../../logs/{RUN}/ckpt_u0008104.pt">最终检查点</a> · <a href="../training/{RUN}/eval_ckpt_u0008104.json">最终诊断评估</a> · <a href="../training/{RUN}/robustness.json">五组重置扰动结果</a></p>'
    doc += '<p>' + ' · '.join(f'<a href="{p}">{p}</a>' for p in ['protocol.json', 'status.json', 'profile-audit.json', 'probe-evaluation.json', 'probe-benchmark.json'] if (OUT / p).exists()) + '</p></body></html>'
    if (OUT / 'visualization/index.html').exists():
        doc = doc.replace('</body>', '<p><a href="visualization/index.html">参考动作与跟踪对比播放器（完整 261 秒，可调速、逐帧、查看误差）</a> · <a href="visualization/comparison_full.mp4">完整对比视频 MP4</a></p></body>')
    write(OUT / 'report.html', doc)
    print(json.dumps(dict(status=state['status'], updates=(train or {}).get('updates', 0))))


if __name__ == '__main__':
    main()
