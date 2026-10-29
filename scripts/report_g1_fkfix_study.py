"""Refresh the repaired G1 experiment's evidence and progress, without GPU work."""
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/g1_fkfix_study'
OLD = 'g1_walk1_e1c_aligned_8104'
NEW = 'g1_walk1_e1c_fkfixed_8104'


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def snapshot(run):
    folder = ROOT / 'reports/training' / run
    result = dict(run_name=run, stage=read(folder / 'status.json'),
                  training=read(ROOT / 'reports/microbatch' / run / 'result.json'),
                  final=read(folder / 'final.json'))
    log = folder / 'training.log'
    result['learning_curve'] = []
    if log.exists():
        lines = [line for line in log.read_text(errors='replace').splitlines() if line.startswith('it ')]
        result['latest_training_log'] = lines[-1] if lines else None
        for line in lines:
            m = re.search(r'it (\d+).*updates (\d+).*train_ret ([\d.eE+-]+).*wm nse1 ([\d.eE+-]+) nse8 ([\d.eE+-]+)', line)
            if m:
                result['learning_curve'].append(dict(iteration=int(m[1]), updates=int(m[2]), train_return=float(m[3]), wm_nse1=float(m[4]), wm_nse8=float(m[5])))
    result['evaluations'] = {str(u): read(folder / f'eval_ckpt_u{u:07d}.json') for u in [2000, 4052, 8104]}
    result['upstream_benchmarks'] = {str(u): read(folder / f'eval_ckpt_u{u:07d}.upstream.json') for u in [2000, 4052, 8104]}
    result['robustness'] = read(folder / 'robustness.json')
    return result


def main():
    OUT.mkdir(exist_ok=True)
    old, new = snapshot(OLD), snapshot(NEW)
    pause_plan = read(ROOT / 'reports/t800_fkfix_study/handoff-plan.json')
    probe = read(ROOT / 'reports/microbatch/g1_fkfixed_fullbatch_policy_probe/result.json')
    state = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
        status=(new['stage'] or {}).get('status', 'preparing'), old=old, corrected=new,
        full_batch_probe=probe, protocol=read(OUT / 'protocol.json'),
        old_upstream_benchmark=read(OUT / 'old-upstream-benchmark.json'),
        requested_pause=pause_plan,
        interpretation='No improvement claim until matched evaluation is available. Old results are affected by the compiled-FK error. No extra anchor, action-scale change, network reduction, or horizon reduction in the corrected G1 run.')
    tmp = OUT / 'status.tmp'
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(OUT / 'status.json')
    esc = html.escape
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 FK 修复对照进度</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#243344}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd7df;padding:9px;text-align:left}th{background:#edf2f7}code{overflow-wrap:anywhere}.note{background:#edf5f1;padding:16px}a{color:#146399}</style><body><h1>G1：修复编译 FK 后的同预算对照</h1>'''
    doc += f'<p>状态：{esc(state["status"])}；更新 UTC：{esc(state["updated_utc"])}</p>'
    if pause_plan:
        doc += '<p class="note"><b>用户已调整执行顺序：</b>本次 G1 在 4052 次检查点完成两种评估后暂停，保留同预算对照，再切换 T800。8104 次是原计划预算，后半程尚未执行。<a href="../t800_fkfix_study/report.html">查看交接与 T800 进度</a>。</p>'
    doc += '<p class="note">固定完整 walk1_subject1（13066 帧，261.3 秒），seed=0、4096 环境、4096 策略 batch、H=32、原学习率计划，训练到 8104 次更新。保留 policy activation recomputation 和 IEEE FP32，仅修正编译 FK 的整数/枚举判断。启动前逐项核对世界模型、策略、critic 和优化器初始状态；不续训旧模型。</p>'
    if probe:
        doc += f'<p>完整 batch 三次更新验证：{esc(probe["status"])}；PyTorch 峰值 {probe["torch_peak_allocated_gib"]:.2f} GiB，整卡采样峰值 {probe.get("device_sampled_peak_used_gib", 0):.2f} GiB。初始状态核对：{esc(str(probe.get("fresh_initialization_checks")))}</p>'
    for label, run in [('原组（编译 FK 错误）', old), ('修复组', new)]:
        training = run['training'] or {}
        doc += f'<p><b>{label}</b>：{training.get("updates", 0)} / 8104；{esc(training.get("status", "pending"))}<br><code>{esc(run.get("latest_training_log") or "尚无训练回报记录")}</code></p>'
    doc += '<table><tr><th>更新次数</th><th>组别</th><th>真实 evaluation return</th><th>完整动作累计归一化回报</th><th>关节 RMSE / rad</th><th>根位置均误差 / m</th><th>完成动作</th></tr>'
    for u in [2000, 4052, 8104]:
        for label, run in [('原组', old), ('修复组', new)]:
            record = run['evaluations'][str(u)]
            if record:
                doc += f'<tr><td>{u}</td><td>{label}</td><td>{record["normalized_return"]:.4f}</td><td>{record["normalized_full_clip_return"]:.4f}</td><td>{record["joint_position_rmse_rad"]:.4f}</td><td>{record["mean_root_position_error_m"]:.3f}</td><td>{record["completed_clip"]}</td></tr>'
            else:
                message = '未保存此检查点' if label == '原组' and u == 2000 else ('按用户指令暂停，尚未执行' if label == '修复组' and u == 8104 and pause_plan else '等待检查点与评估')
                doc += f'<tr><td>{u}</td><td>{label}</td><td colspan="5">{message}</td></tr>'
    doc += '</table><p>evaluation return 是实际执行步的平均奖励 / 0.1；完整动作累计归一化回报将未执行帧计零。提前失败时两者不可混用。上表原组使用既有物理诊断评估，修复组使用固定上游源码加同口径只读诊断；另行执行未经修改的上游 benchmark，作为与作者 e1c（0.676）的正式比较。</p>'
    diagnostic=new['evaluations'].get('4052')
    if diagnostic and not diagnostic['completed_clip']:
        doc+=f'<p class="note">修复组 4052 次的 IEEE 诊断回放在 {diagnostic["steps"]}/13066 帧因 {esc(str(diagnostic["termination_reasons"]))} 提前终止；其平均回报不能直接代表完整动作表现。同一检查点的原版上游 benchmark 使用默认 TF32，此次完整完成。此差异限制了对单次回放稳定性的推断。</p>'
    doc += '<table><tr><th>上游 benchmark</th><th>Normalized return</th><th>步数 / 总帧数</th><th>失败</th></tr>'
    benchmarks = [('原组 8104', state['old_upstream_benchmark'])]
    old_mid=read(ROOT/'reports/t800_fkfix_study/g1-old-4052.upstream.json')
    if old_mid:benchmarks.insert(0,('原组 4052',old_mid))
    benchmarks += [(f'修复组 {u}', new['upstream_benchmarks'][str(u)]) for u in [2000, 4052, 8104]]
    for label, record in benchmarks:
        if record:
            c = record['clips'][0]
            doc += f'<tr><td>{label}</td><td>{c["norm_return"]:.4f}</td><td>{c["steps"]} / {c["frames"]}</td><td>{c["failed"]}</td></tr>'
        else:
            doc += f'<tr><td>{label}</td><td colspan="3">等待评估</td></tr>'
    doc += '</table><p>自动保存并评估 2000、4052、8104 次检查点；评估不更新网络、不消耗训练随机数。若并发评估失败，训练继续，评估在训练结束后重试。最终增加 5 组重置扰动；单次训练、单一 seed，不能代表多次训练的统计结论。</p>'
    links = [('protocol.json', '冻结协议'), ('status.json', '完整进度与指标'), ('../g1_convergence_audit/report.html', '问题定位与修复证据'), (f'../training/{NEW}/status.json', '后台阶段状态')]
    doc += '<p>' + ' · '.join(f'<a href="{p}">{title}</a>' for p, title in links if (OUT / p).exists()) + '</p></body></html>'
    (OUT / 'report.html').write_text(doc)
    print(json.dumps(dict(status=state['status'], updates=(new['training'] or {}).get('updates', 0))))


if __name__ == '__main__':
    main()
