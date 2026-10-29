"""Refresh the status and evidence index without modifying any training process."""
from datetime import datetime,timezone
import hashlib
import html
import json
from pathlib import Path
import re

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/task_alignment_study'
RUNS=[('g1','g1_walk1_e1c_aligned_8104','G1 · 完整 walk1'),
      ('t800_full','t800_walk1_full_aligned_8104','T800 · 完整 walk1'),
      ('t800_crop','t800_walk1_crop_aligned_8104','T800 · 同源截取')]


def read(p):
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    r=dict(updated_utc=datetime.now(timezone.utc).isoformat(),status='in_progress',stages={},
        data_protocol=read(OUT/'data-protocol.json'),adapter_audit=read(OUT/'task-adapter-audit.json'),
        reference_quality=read(OUT/'reference-quality-status.json'),adaptation_review=read(OUT/'adaptation-review.json'),
        runtime_compatibility=read(OUT/'compatibility-blocker.json'),
        upstream_reference=dict(experiment='e1c',updates=8104,full_frames=13066,train_return_approx=.78,eval_return=.676),
        note='No claim of reproduction success until the full benchmark has completed.')
    rows=[]
    for key,run,label in RUNS:
        stage=read(ROOT/'reports/training'/run/'status.json')
        train=read(ROOT/'reports/microbatch'/run/'result.json')
        final=read(ROOT/'reports/training'/run/'final.json')
        record=dict(run_name=run,label=label,stage=stage,training=train,final=final)
        log=ROOT/'reports/training'/run/'training.log'
        if log.exists():
            lines=[x for x in log.read_text(errors='replace').splitlines() if x.startswith('it ')]
            record['latest_training_log']=lines[-1] if lines else None
            history=[]
            for line in lines:
                m=re.search(r'it (\d+).*updates (\d+).*train_ret ([\d.eE+-]+).*wm nse1 ([\d.eE+-]+) nse8 ([\d.eE+-]+)',line)
                if m:history.append(dict(iteration=int(m[1]),updates=int(m[2]),train_return=float(m[3]),wm_nse1=float(m[4]),wm_nse8=float(m[5])))
            record['learning_curve']=history
        state=stage['status'] if stage else 'pending'
        updates=train.get('updates',0) if train else 0
        r['stages'][key]=record
        rows.append(f'<tr><td>{label}</td><td>{state}</td><td>{updates} / 8104</td><td>{final["normalized_return"]:.4f}</td></tr>' if final else f'<tr><td>{label}</td><td>{state}</td><td>{updates} / 8104</td><td>尚未完成</td></tr>')
    benchmark=read(OUT/'g1-upstream-benchmark.json')
    r['g1_upstream_benchmark']=benchmark
    if benchmark:
        c=benchmark['clips'][0]
        r['g1_comparison']=dict(return_difference=c['norm_return']-.676,completed=not c['failed'] and c['steps']==13066,
            comparability_flag='close' if not c['failed'] and abs(c['norm_return']-.676)<=.05 else 'gap_requires_review',
            flag_definition='Descriptive ±0.05 band from one run, not a statistical equivalence test.')
    cross=read(OUT/'cross-evaluations.json');r['cross_evaluations']=cross
    if all(x['stage'] and x['stage']['status']=='finished' for x in r['stages'].values()) and benchmark and cross and cross.get('status')=='finished':
        r['status']='completed'
    if r['runtime_compatibility'] and r['runtime_compatibility'].get('status')=='confirmed':
        r['status']='runtime_bug_confirmed'
        r['note']='The original G1 run is affected by the confirmed compiled-FK error. Its results do not establish a valid upstream reproduction. The corrected run uses a separate execution protocol; see reports/g1_fkfix_study/report.html.'
    tmp=OUT/'status.tmp';tmp.write_text(json.dumps(r,indent=2,ensure_ascii=False)+'\n');tmp.replace(OUT/'status.json')
    mapping=''
    if r['adapter_audit']:
        mp=r['adapter_audit']['task_adapter']['g1_to_t800']
        mapping='<table><tr><th>上游 G1</th><th>T800 对应刚体</th></tr>'+''.join(f'<tr><td>{a}</td><td>{b}</td></tr>' for a,b in mp.items())+'</table>'
    logs=''.join(f'<p><b>{x["label"]}</b><br><code>{html.escape(x.get("latest_training_log") or "尚无训练日志")}</code></p>' for x in r['stages'].values())
    summary=''
    if benchmark:
        c=benchmark['clips'][0]
        summary=f'<p>原版上游 benchmark：return <b>{c["norm_return"]:.4f}</b>，执行 {c["steps"]}/{c["frames"]} 步，失败 {c["failed"]}。上游参考为 0.676、13066 步完整完成。</p>'
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>SuperMimic 任务对齐实验</title><style>body{font:16px/1.6 system-ui,sans-serif;max-width:1100px;margin:40px auto;padding:0 24px;color:#243344}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border:1px solid #ccd7df;padding:8px;text-align:left}th{background:#eaf1f7}code{font-size:13px;overflow-wrap:anywhere}.notice{background:#edf5f1;padding:16px}a{color:#185f91}</style><body><h1>SuperMimic：完整任务复现与 T800 适配</h1>'''
    doc+=f'<p>状态：{r["status"]} · 更新 UTC：{r["updated_utc"]}</p>'
    if r['runtime_compatibility']:
        doc+='<div style="background:#fff1e7;border-left:5px solid #bf5d28;padding:16px"><b>执行错误已确认：</b>当前 Torch 编译路径错误跳过 FK 的铰链关节分支，G1/T800 均受影响。以下保留原实验记录；源码和配置一致不能代表运行正确。<a href="../g1_convergence_audit/report.html">查看定位与验证报告</a>；<a href="../g1_fkfix_study/report.html">查看另行启动的修复对照</a>。</div>'
    doc+='''<div class="notice">先完成 G1 的完整 walk1 复现，再评估同动作的 T800 适配及同源截取对照。当前报告随训练进度更新；尚未完成的指标不代表最终结果。</div><table><tr><th>阶段</th><th>状态</th><th>策略更新</th><th>最终 evaluation return</th></tr>'''+''.join(rows)+'</table>'+summary+logs
    doc+='''<h2>固定的比较条件</h2><p>上游 e1c：walk1_subject1 全部 13066 帧（261.3 秒），无额外 9 维 anchor 观测。4096 环境、4096 策略 batch、H=32、UTD=0.4，原 320000 次环境迭代学习率计划，实际停止在 8104 次策略更新。</p><p>本机采用经过对拍的 policy activation recomputation 和 IEEE FP32，保持完整有效批量、网络和 horizon。RTX 5070 Ti / PyTorch 2.11 cu128 与上游 RTX 5090 / PyTorch 2.14 cu130 有差异，不能宣称位级复现。</p>
<h2>T800 适配</h2><p>保留已验证的 T800 模型、25 个受控关节、Kp/Kd、力矩限制、默认姿态及碰撞。动作尺度使用 0.25 × effort_limit / Kp。reward anchor 改为上身的 LINK_WAIST_YAW；根状态及摔倒判定仍使用 LINK_BASE。跟踪 14 个对应刚体，头部继续受控，但不额外加入刚体奖励平均。肘部使用屈伸关节处的 ELBOW_PITCH，而非下游的 ELBOW_YAW。</p>'''+mapping
    doc+='''<h2>完整动作与截取对照</h2><p>T800 完整 walk1 先连续 IK，再整体计算速度和 FK。已在训练前完成初始 IK 姿态筛选：原初值导致右肩长期贴限位，采用新的固定初值后该现象消失。使用 t800_walk1_selected_seed / t800_walk1_selected_crop 数据，原始版本保留用于审计。未修改 IK 目标权重，也未逐帧重启或平滑参考。</p><p>短片段固定截取输出第 500–698 帧（10 秒起、199 帧），不重新 IK、不重新计算边界速度、不平移世界坐标。六组参考数组逐元素相同；两组训练采用相同任务适配、初始网络和更新预算。</p><p>完整模型还需在同一短片段上评估，再与短片段模型比较。不同训练数据分布、重置频率和片段末端处理是本组对照共同改变的因素，不能凭单次对照把它们彼此分离。</p>
<h2>补充适配检查</h2><p><a href="adaptation-review.html">查看 T800 适配补充报告</a>：新参考状态、执行器、奖励覆盖、CPU 局部动态检查，以及专用训练入口与旧默认配置的区别。参考质量检查不是训练 evaluation，也不证明全动作动态可行。</p>
<h2>证据文件</h2><ul><li><a href="g1-protocol.json">G1 冻结协议</a></li><li><a href="data-protocol.json">同源动作与截取验证</a></li><li><a href="task-adapter-audit.json">刚体映射与奖励对拍</a></li><li><a href="reference-quality-status.json">训练前 IK 初值与参考质量检查</a></li><li><a href="t800-selected-motion-audit.json">选定完整动作几何检查</a></li><li><a href="t800-selected-crop-motion-audit.json">选定截取动作几何检查</a></li><li><a href="status.json">结构化进度和结果</a></li></ul></body></html>'''
    (OUT/'report.html').write_text(doc)
    print(json.dumps(dict(status=r['status'],progress={k:(x['training'] or {}).get('updates',0) for k,x in r['stages'].items()})))


if __name__=='__main__': main()
