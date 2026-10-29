"""Publish the confirmed compiled-FK diagnosis and its experimental limits."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/g1_convergence_audit'


def read(name):
    p=OUT/name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    bad=read('compiled-step.json');fixed=read('compiled-step-fixed.json')
    source=read('source-config-audit.json')
    regressions={r:read(f'{r}-fk-regression.json') for r in ['g1','t800']}
    parity=read('ckpt4052-recompute-parity.json')
    fixed_parity=read('ckpt4052-fixed-recompute-parity.json')
    control=read('ckpt4052-fixed-original-control.json')
    policy=read('ckpt4052-physical.json');public=read('public-e8-physical.json')
    probe_path=ROOT/'reports/microbatch/g1_fkfixed_fullbatch_policy_probe/result.json'
    probe=json.loads(probe_path.read_text()) if probe_path.exists() else None
    fixed_stage_path=ROOT/'reports/training/g1_walk1_e1c_fkfixed_8104/status.json'
    fixed_stage=json.loads(fixed_stage_path.read_text()) if fixed_stage_path.exists() else None
    assert all(x['passed'] for x in regressions.values())
    b=bad['comparisons']['ieee'];f=fixed['comparisons']['ieee']
    assert f['outputs']['reward']['max_abs']<2e-6
    assert f['reward_gradient']['action']['relative_l2']<1e-4
    findings=dict(
        updated_utc=datetime.now(timezone.utc).isoformat(),status='confirmed_runtime_compilation_bug',
        primary_cause='torch.compile on this PyTorch 2.11 installation folds Python int == MuJoCo pybind enum to false in FK.forward, omitting all hinge rotations and joint-velocity contributions.',
        affected_robots=['g1','t800'],
        affected_path='Compiled ImagStep -> AnalyticHeads.robot_bodies -> G1Kinematics.forward',
        unaffected_paths=['Eager FK validation','CPU reference retargeting and conversion','Real MuJoCo environment reward computation','World-model state transition loss'],
        source_config_audit=source,
        minimal_reproducer=regressions['g1']['minimal_reproducer'],
        before_after=dict(reward_max_abs=[b['outputs']['reward']['max_abs'],f['outputs']['reward']['max_abs']],
            action_reward_gradient_cosine=[b['reward_gradient']['action']['cosine'],f['reward_gradient']['action']['cosine']],
            action_reward_gradient_relative_l2=[b['reward_gradient']['action']['relative_l2'],f['reward_gradient']['action']['relative_l2']]),
        regressions=regressions,
        memory_recompute=dict(original_compiled_path_parity_passed=parity['passed'],
            original_compiled_path_scope=parity['scope'],
            corrected_path_strict_pass=fixed_parity['passed'] if fixed_parity else None,
            corrected_original_control_strict_pass=control['passed'] if control else None,
            corrected_control_comparisons_identical=(control['comparisons']==fixed_parity['comparisons']) if control and fixed_parity else None,
            corrected_control_failure_names_identical=([x['name'] for x in control['failures']]==[x['name'] for x in fixed_parity['failures']]) if control and fixed_parity else None,
            note='Original-versus-recompute agreement alone did not test correctness: both used the same broken compiled FK. After correction, all 324 comparison statistics and all 18 strict failure names match the original-versus-original control exactly. These observed critic differences are therefore not specific to recomputation. This is a batch-64, two-update check, not full-batch or long-run equivalence.'),
        physical_positive_control=dict(current_checkpoint_updates=4052,
            current_returns=[w['normalized_return'] for w in policy['windows']],
            published_e8_returns=[w['normalized_return'] for w in public['windows']],
            current_joint_rmse=[w['joint_rmse_rad'] for w in policy['windows']],
            published_e8_joint_rmse=[w['joint_rmse_rad'] for w in public['windows']],
            scope='Four identical 5-second windows. Public e8 has a different training dataset/budget and is an environment sanity control, not an e1c reproduction baseline.'),
        correction=dict(helper='scripts/fk_compile_compat.py',training_entrypoint='scripts/train_memory_fk_compat.py',
            helper_sha256=hashlib.sha256((ROOT/'scripts/fk_compile_compat.py').read_bytes()).hexdigest(),
            launcher_sha256=hashlib.sha256((ROOT/'scripts/train_memory_fk_compat.py').read_bytes()).hexdigest(),
            vendor_files_modified=False,active_training_modified=False,
            checkpoint_metadata='execution.fk_compile_compat; corrected launcher rejects resume without matching compatibility metadata',
            full_batch_memory_after_fix_tested=bool(probe and probe.get('status')=='passed'),
            full_batch_probe='reports/microbatch/g1_fkfixed_fullbatch_policy_probe/result.json',
            corrected_training_started=bool(fixed_stage),
            corrected_study='reports/g1_fkfix_study/report.html'),
        interpretation=[
            'Confirmed wrong objective geometry and gradients are the first issue to fix; this is stronger evidence than hypothesizing action-scale or anchor problems.',
            'Small state NSE is compatible with wrong analytic reward geometry. The state predictor itself need not be inaccurate.',
            'Historical T800 anchor/action-scale conclusions apply only to the faulty compiled execution and cannot establish their behavior under the corrected algorithm.',
            'No experiment here proves that this fix alone reproduces the author\'s final e1c return; a fresh controlled training run is still required.',
            'PyTorch 2.14 behavior was not tested locally; this report demonstrates the bug on the installed 2.11 stack, without assuming which newer version fixes it.',
        ],
        prior_validation_gap='Earlier tests covered eager FK versus MuJoCo and compared two implementations sharing the same compiled ImagStep. They did not compare compiled FK against the independent physical oracle.',
        next_steps=['Preserve the existing run as the baseline affected by the compiler bug.',
            'Require corrected full-batch memory validation (the restored FK graph is larger).',
            'Fresh G1 e1c-budget training with the compatibility fix, unchanged task/networks/optimizer/batch/horizon; do not treat old critic/actor resume as a clean reproduction.',
            'Require compiled FK/gradient regression for T800 before further anchor/action-scale conclusions.'])
    (OUT/'diagnosis.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2)+'\n')
    blocker=dict(status='confirmed',issue='compiled_fk_hinge_enum_comparison',robots=['g1','t800'],
        evidence='../g1_convergence_audit/diagnosis.json',
        message='Old protocol source/config agreement does not establish runtime correctness. Compiled FK omits hinge motion on the current Torch stack. New training requires the compatibility fix and a new execution protocol.',
        active_g1_training_left_running_as_requested=True)
    (ROOT/'reports/task_alignment_study/compatibility-blocker.json').write_text(json.dumps(blocker,indent=2)+'\n')
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>G1 收敛差距：编译 FK 错误定位</title>
<style>body{font:16px/1.7 system-ui,sans-serif;max-width:1080px;margin:36px auto;padding:0 24px;color:#243344}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #c8d4df;padding:10px;text-align:left}th{background:#edf2f7}pre{background:#f1f4f7;padding:15px;overflow:auto}.critical{background:#fff1e7;border-left:5px solid #bf5d28;padding:16px}a{color:#145f95}code{overflow-wrap:anywhere}</style><body>
<h1>G1 与 e1c 的差距：已确认编译 FK 兼容错误</h1>'''
    doc+=f'<p>更新 UTC：{html.escape(findings["updated_utc"])}</p>'
    if fixed_stage:
        doc+='<p class="critical"><b>后续实验已启动：</b>修复后的完整 batch 显存检查已通过，同初始条件的 G1 8104 次对照正在执行。<a href="../g1_fkfix_study/report.html">查看修复对照进度和结果</a>。下文为问题定位证据。</p>'
    doc+='''<div class="critical"><b>核心结论：</b>当前源码基本沿用固定版本上游，但在本机 PyTorch 2.11 的 torch.compile 路径中，铰链关节分支被错误跳过。想象奖励计算使用了错误的身体姿态和速度，策略得到错误梯度。G1 与 T800 都受影响。这是已复现的执行错误，尚不等于修复后已复现作者最终成绩。</div>
<h2>具体错在哪里</h2><p>可微 FK 将关节类型存成 Python 整数，forward 中用它与 MuJoCo 的 pybind 枚举比较。非编译时比较为真，当前编译路径却将它折叠为假：</p>
<pre>if l.joint_type == mujoco.mjtJoint.mjJNT_HINGE:
    # 应计算关节旋转与关节速度贡献</pre>
<p>独立最小复现中，本应返回 2 的同一个函数，编译后返回 3；加上显式 int 转换后，两种路径均返回 2。使用 Dynamo 的 eager backend 就能复现，说明不需要 CUDA 或 Inductor 的浮点优化便会出错。</p>
<pre>if l.joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
    # 保留原来的所有计算，只明确整数比较</pre>
<h2>修复前后证据</h2><p>从 4052 次检查点的真实仿真回放取 64 个状态，对同一模型、同一输入比较 eager 与 GPU 编译结果：</p>
<table><tr><th>指标</th><th>修复前</th><th>修复后（IEEE FP32）</th></tr>'''
    doc+=f'<tr><td>单步奖励最大绝对差</td><td>{b["outputs"]["reward"]["max_abs"]:.6f}</td><td>{f["outputs"]["reward"]["max_abs"]:.2e}</td></tr>'
    doc+=f'<tr><td>奖励对动作梯度的方向余弦</td><td>{b["reward_gradient"]["action"]["cosine"]:.5f}</td><td>{f["reward_gradient"]["action"]["cosine"]:.10f}</td></tr>'
    doc+='''<tr><td>独立 FK 对 MuJoCo，G1/T800 各 16 个随机姿态</td><td>位置最大差可超过 1 m；身体位置对关节角的梯度范数为 0</td><td>float64 姿态/速度最大差小于 10⁻¹⁴；梯度恢复</td></tr></table>
<p>这里的随机姿态测试用于检查函数是否正确，并非训练策略的实际误差。TF32 开关都能复现错误；修复后两种精度均恢复一致，因此 TF32 不是这个分支错误的根因。</p>
<h2>我们修改了什么，哪些没有改</h2><ul>
<li>G1 实际加载 <code>vendor/supermimic_upstream/src</code>。训练算法、模型、动作映射和 G1 物理配置未混入 T800 适配；算法配置与固定源码默认值逐项一致。</li>
<li>为节省显存，用非重入 activation checkpointing 重算策略网络激活；4096 环境、4096 策略 batch、H=32、网络规模、GAE、优化器及 critic 更新次数保留。当前没有使用 microbatch 训练。</li>
<li>执行环境为 PyTorch 2.11/cu128，与作者文档中的 2.14/cu130 不同；我们使用 IEEE FP32，作者默认 TF32。还有检查点、停止条件和日志包装。</li>
<li>4052 次检查点中，WM/actor/critic 的 Adam 更新次数分别为 40520/4052/81040，学习率约 1.992×10⁻⁴；没有发现少更新、过早学习率衰减或非有限参数。</li></ul>
<h2>为什么之前的检查没发现</h2><p>此前 FK 对 MuJoCo 的检查走的是非编译路径；原版与重计算的对拍，则两边都走同一个错误的编译 FK。它们一致，不能证明奖励计算正确。此前将源码/配置对齐等同于运行正确的判断过于乐观。本次补上了“编译计算与独立物理参考”的检查。</p>
<p>修复前，两次完整 32 步更新的原版/重计算对拍通过。修复后的 actor 梯度差仍很小；20 次 critic 更新的严格逐元素对拍另有浮点差异，详细控制结果如下，不把它隐藏为全部通过。</p>'''
    doc+=f'<pre>{html.escape(json.dumps(findings["memory_recompute"],ensure_ascii=False,indent=2))}</pre>'
    if findings['memory_recompute']['corrected_control_comparisons_identical']:
        doc+='<p>控制实验已完成：修复后，原版对原版与原版对重计算的全部 324 项误差统计完全相同，18 项严格阈值失败也相同。因此这些 critic 数值差异不是本次显存重计算独有的问题。测试使用 batch=64、两次更新，不能据此宣称完整 batch 或长训练完全等价。</p>'
    doc+='''<h2>同一环境的预训练模型控制</h2><p>作者公开的 e8 模型与我们的 4052 次模型，在完全相同的四个 5 秒窗口中：</p><table><tr><th>起始帧</th><th>本地 4052 次 return / 关节 RMSE</th><th>作者 e8 return / 关节 RMSE</th></tr>'''
    for a,z in zip(policy['windows'],public['windows']):
        doc+=f'<tr><td>{a["start_frame"]}</td><td>{a["normalized_return"]:.3f} / {a["joint_rmse_rad"]:.3f} rad</td><td>{z["normalized_return"]:.3f} / {z["joint_rmse_rad"]:.3f} rad</td></tr>'
    doc+='''</table><p>e8 模型能在当前环境中表现良好，说明环境/参考/实际奖励链路支持高质量跟踪。e8 的数据和预算不同，这不是与 e1c 的同预算结果比较，也不是完整动作 benchmark。</p>
<h2>对既有 T800 结论的影响</h2><p>同一枚举分支错误在 T800 中也复现。状态预测 NSE 可以很小，而解析奖励仍是错的；仅靠继续训练、加 anchor 或改 action scale 无法修复这个错误。旧实验的实际 evaluation 数值仍是真实测量，但关于这些因素有效或无效的结论，需要在修复后重新验证。</p>
<h2>交付与下一步</h2><p>已提供独立、可记录版本的兼容修复 <code>scripts/fk_compile_compat.py</code>，以及新训练入口 <code>scripts/train_memory_fk_compat.py</code>。新检查点记录 <code>execution.fk_compile_compat</code>，拒绝将未标记的旧检查点作为同协议续训。未修改 vendor 文件，未向旧 G1 进程注入修复。</p><p>原 G1 作为受该错误影响的基线保存。正确复现须先验证修复后的完整 batch 显存，再用相同任务、8104 更新预算从头训练。本报告的函数及梯度验证没有证明最终回报；后续训练状态以独立的修复对照报告为准。</p>
<h2>证据文件</h2><ul>'''
    names=['source-config-audit.json','g1-fk-regression.json','t800-fk-regression.json','compiled-step.json','compiled-step-fixed.json',
           'ckpt4052-recompute-parity.json','ckpt4052-fixed-recompute-parity.json','ckpt4052-fixed-original-control.json',
           'ckpt4052-physical.json','public-e8-physical.json','diagnosis.json','upstream-source-history.json']
    doc+=''.join(f'<li><a href="{n}">{n}</a></li>' for n in names if (OUT/n).exists())
    doc+='</ul></body></html>'
    (OUT/'report.html').write_text(doc)
    print(json.dumps(dict(status=findings['status'],before_after=findings['before_after'],memory_recompute=findings['memory_recompute']),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
