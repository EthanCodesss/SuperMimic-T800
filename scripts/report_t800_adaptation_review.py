"""Summarize the extra adaptation audit while leaving active training unchanged."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/task_alignment_study"


def read(name):
    return json.loads((OUT / name).read_text())


def main():
    t = read("t800-extended-audit.json")
    g = read("g1-extended-audit.json")
    historical = read("historical-head-error-audit.json")
    blocker_path = OUT / "compatibility-blocker.json"
    blocker = json.loads(blocker_path.read_text()) if blocker_path.exists() else None
    protocol_checks = {}
    for filename in ("g1-protocol.json", "t800-full-protocol.json", "t800-crop-protocol.json"):
        p = read(filename)
        bad = [name for name, expected in p["frozen_file_sha256"].items()
               if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected]
        assert not bad, (filename, bad)
        protocol_checks[filename] = dict(frozen_files_verified=True, files=len(p["frozen_file_sha256"]))
    result = dict(
        updated_utc=datetime.now(timezone.utc).isoformat(),
        conclusion="No additional mapping/actuator/reference-state inconsistency found in sampled checks. Head reward coverage and generic-entrypoint defaults remain explicit adaptation limitations; full dynamic feasibility and convergence are not established.",
        protocol_checks=protocol_checks,
        confirmed_findings=[
            dict(id="ik_branch", severity="reference_issue_corrected_before_training",
                 detail="Original full walk1 IK followed an undesirable right-arm branch; the selected initial seed improves full-motion fit and removes persistent right-shoulder hard-limit occupancy.",
                 evidence="reference-quality-status.json", explains_historical_short_walk2=False),
            dict(id="head_coverage", severity="new_profile_design_gap",
                 detail="The aligned 14-body task retains 25 action joints but excludes the head from body tracking rewards. Perturbing either head joint by 0.2 rad produces zero direct tracking-reward change; both reference head joints are constant zero.",
                 historical_baseline_head_squared_error_share=historical["t800_short_noanchor_8000"]["head_squared_error_share"],
                 explains_historical_short_error=False,
                 action="Retain frozen experiment conditions; report non-head and head tracking separately. Any head fixation or head-specific reward belongs to an explicit subsequent profile."),
            dict(id="entrypoint", severity="configuration_hazard",
                 detail="Generic T800 source defaults retain LINK_BASE, 15 tracked bodies and RL Lab action scale. train_task_aligned.py/evaluate_task_aligned.py install the 14-body waist-anchor torque/Kp profile and validate checkpoint metadata.",
                 correct_stage="scripts/run_task_aligned_stage.py", correct_evaluator="scripts/evaluate_task_aligned.py",
                 action="Use the protocol-specific entrypoint for the next T800 experiment; original checkpoints keep their original evaluator."),
        ],
        validated=[
            "25-joint names/order, actuator torque and gain/armature tables, and absence of unintended raw-action [-1,1] clipping.",
            "Reference/model body transforms, mass/inertia, joint axes and limits; saved body positions, quaternions and velocities agree with independently evaluated MuJoCo states on 512 frames.",
            "Existing aligned GPU audit: analytic reward, observed newest frame and future goals agree with environment computation.",
            "New reference has no joint speed above configured nominal ratings; local PD windows have no T800 torque saturation.",
            "Contact and constraint counts in the sampled states/windows stay below configured capacities; this does not cover all policy states.",
            "Core source locks and all three experiment protocol hashes remain unchanged.",
        ],
        comparisons=dict(
            t800_reference_state_max_abs_error=max(t["reference_body_state_max_errors"].values()),
            t800_ideal_reference_reward_with_reference_pd_action_rate=t["reference_ideal_normalized_reward_with_reference_pd_action_rate"],
            g1_ideal_reference_reward_with_reference_pd_action_rate=g["reference_ideal_normalized_reward_with_reference_pd_action_rate"],
            t800_supported_root_joint_rmse_rad=t["local_pd_rollouts"]["reference_root_prescribed"]["combined_joint_rmse_rad"],
            g1_supported_root_joint_rmse_rad=g["local_pd_rollouts"]["reference_root_prescribed"]["combined_joint_rmse_rad"],
            t800_free_root_pd_1s_completion_fraction=t["local_pd_rollouts"]["free_root"]["completed_fraction"],
            g1_free_root_pd_1s_completion_fraction=g["local_pd_rollouts"]["free_root"]["completed_fraction"],
        ),
        unresolved=[
            "G1 full benchmark has not yet completed; shared optimization/execution/convergence issues cannot be separated from T800 adaptation effects yet.",
            "Low world-model state NSE does not certify long-horizon action-gradient or critic accuracy.",
            "Reference kinematics and local PD tests do not establish free-base full-motion feasibility.",
            "RL Lab USD/PhysX dynamics equivalence remains unverified. Current port uses GMR/SDK MJCF and RL Lab gains; randomized actuator delay and hard velocity caps are absent by design.",
            "Some ankle reference frames still violate soft limits; their mean ideal normalized reward penalty is about 0.0009, insufficient by itself to explain a 0.5 return.",
        ],
        no_training_or_frozen_recipe_modifications=True,
        evidence=["t800-extended-audit.json", "g1-extended-audit.json", "task-adapter-audit.json",
                  "gmr-sdk-model-comparison.json", "historical-head-error-audit.json", "reference-quality-status.json"],
    )
    result["runtime_compatibility"] = blocker
    if blocker:
        result["conclusion"] = "The mapping/actuator/reference checks below remain valid within their sampled eager/physical scope. A subsequent audit confirmed a shared compiled-FK bug affecting G1 and T800 policy optimization; see ../g1_convergence_audit/report.html. Runtime correctness and clean convergence are not established by these earlier checks."
        result["confirmed_findings"].insert(0, dict(id="compiled_fk", severity="confirmed_training_bug", detail=blocker["message"], evidence="../g1_convergence_audit/diagnosis.json"))
        result["unresolved"][0] = "A corrected full-budget G1 reproduction remains pending. The currently running G1 baseline uses the faulty compiled FK."
    (OUT / "adaptation-review.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    e = html.escape
    c = result["comparisons"]
    doc = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T800 适配补充检查</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#253346;max-width:1050px;margin:36px auto;padding:0 24px}table{width:100%;border-collapse:collapse}td,th{padding:10px;border:1px solid #cbd5df;text-align:left}th{background:#edf2f7}code{overflow-wrap:anywhere}.note{padding:16px;background:#edf5f0}a{color:#12609a}</style><body>
<h1>T800 适配补充检查</h1>'''
    doc += f'<p>更新 UTC：{e(result["updated_utc"])}</p>'
    if blocker:
        doc += '<p style="background:#fff1e7;border-left:5px solid #bf5d28;padding:16px"><b>后续检查发现关键执行错误：</b>G1 与 T800 的编译 FK 均错误跳过铰链关节分支，导致想象奖励和梯度错误。下文保留较早的接口及物理检查，其解析奖励检查没有覆盖这条编译路径。后续训练必须先接入兼容修复并更新执行协议，不能直接沿用旧入口开始新对照。<a href="../g1_convergence_audit/report.html">完整定位与修复报告</a>。</p>'
    doc += '''<p class="note">本轮未发现新的关节错位、PD 参数写错或参考状态与模型不一致。已修复的完整 walk1 IK 分支问题仍是明确的数据问题；新任务的头部奖励覆盖和训练入口差异需要记录。G1 后台训练、冻结算法和 T800 训练协议均未改动。</p>
<h2>确认的问题与边界</h2><ol>
<li><b>完整 walk1 的 IK 初值问题已修复。</b>旧分支右肩长期贴限位，新分支右手目标误差由 16.1 cm 降到 6.6 cm。它发生在这条完整 walk1 的生成中，不能直接解释旧 walk2 短动作训练结果。</li>
<li><b>新 14 刚体任务没有直接跟踪头部。</b>动作仍有 25 维，但 head pitch/yaw 扰动 0.2 rad 对直接姿态跟踪奖励没有影响。头部仍受动力学、软限位和动作变化惩罚影响。参考头关节恒为零；这不是旧实验回报低的解释，因为旧任务包含头部奖励，而且旧无 anchor 基线头部仅占总关节平方误差的 0.63%。</li>
<li><b>新适配需要专用入口。</b>普通 T800 入口保留原来的 BASE anchor、15 刚体和固定 action scale。下一组对照必须使用 <code>scripts/run_task_aligned_stage.py</code>，评估使用 <code>scripts/evaluate_task_aligned.py</code>；检查点带任务与动作尺度元数据，专用评估器会拒绝不匹配的模型。</li></ol>
<h2>同一方法下的 G1 / T800 检查</h2><table><tr><th>指标</th><th>T800 新参考</th><th>G1 参考</th></tr>'''
    rows = [
        ("理想参考 reward（含软限位、相邻参考 PD 动作变化）", c["t800_ideal_reference_reward_with_reference_pd_action_rate"], c["g1_ideal_reference_reward_with_reference_pd_action_rate"]),
        ("每个子步恢复根状态到参考：关节 RMSE / rad", c["t800_supported_root_joint_rmse_rad"], c["g1_supported_root_joint_rmse_rad"]),
        ("自由根部：直接参考 PD 的 1 秒窗口完成率", c["t800_free_root_pd_1s_completion_fraction"], c["g1_free_root_pd_1s_completion_fraction"]),
    ]
    for label, tv, gv in rows:
        doc += f'<tr><td>{label}</td><td>{tv:.5f}</td><td>{gv:.5f}</td></tr>'
    doc += '''</table><p>后两项在各动作均匀选取 32 个起点，使用 CPU MuJoCo、50 Hz 参考角度目标和 200 Hz PD 仿真，非策略 evaluation。根状态恢复测试在每个 5 ms 子步开始时写入插值参考根状态，提供人为支撑，不是一个物理平衡控制器；自由根部测试没有平衡策略或动力学补偿。G1 同样容易偏离，因此不能用直接参考 PD 的失败判定 T800 参考不可学习。</p>
<h2>为什么 PD 目标不必等于参考角度</h2><p>位置 PD 的力矩为 Kp × (目标角度 − 实际角度) − Kd × 实际角速度。即使实际姿态精确跟随参考，也通常需要目标角度偏置来提供重力、惯性和阻尼补偿。本轮按完整参考运动做无接触逆动力学，T800 肘部所需偏置绝对值的中位数约 0.022–0.023 rad、99 分位约 0.068 rad。此估计仅适用于无外部手部接触且根部运动已给定的条件；不能用于判断腿部接触可行性，也不能解释旧动作上数倍于此的持续偏差。</p>
<h2>已经排查的接口</h2><ul>
<li>参考的 25 个关节顺序、身体名称、四元数次序和角速度坐标转换；512 帧独立 FK/速度核对最大绝对误差小于 6×10⁻⁷。</li>
<li>模型构建前后的关节轴、刚体变换、质量和惯量一致；GMR 与已固定版本的 SDK XML 中 29 个子刚体、25 个关节、52 个几何体的所查属性相同。</li>
<li>实际编译后的 Kp/Kd、力矩上限、armature 与 T800 配置一致；动作目标没有被错误限制到 [-1,1]。</li>
<li>新参考最大关节速度约 10.5 rad/s，各关节均低于各自配置中的标称速度；本轮局部 T800 PD 测试没有力矩饱和。</li>
<li>既有新任务 GPU 检查中，环境与解析奖励最大误差约 7.2×10⁻⁷，最新观测约 8.9×10⁻⁷，未来目标完全一致。</li></ul>
<h2>仍未排除的原因</h2><p>优先完成 G1 的完整 benchmark，以判断当前优化与执行环境是否能达到同任务的上游效果；然后按冻结条件训练新 T800 参考。世界模型状态误差小尚不能排除策略梯度、critic 或长期滚动偏差。也不能从关节伺服通过推出全身接触动态可行。当前没有证据支持继续盲调 action scale、加大 PD 或立即改奖励。</p>
<p>MuJoCo 模型与 RL Lab USD/PhysX 的完整动态等价性尚未验证；当前按既定方案省略随机执行延迟，也没有硬速度截断。新参考未超过标称速度，暂未发现这两项是本任务误差的直接来源。</p>
<h2>证据与复现</h2><p>运行 <code>scripts/audit_adaptation_extended.py --robot t800|g1 --motion 对应NPZ --out 结果JSON</code> 可复现本轮 CPU 诊断。</p><ul>'''
    doc += ''.join(f'<li><a href="{e(n)}">{e(n)}</a></li>' for n in result["evidence"])
    doc += '<li><a href="adaptation-review.json">结构化检查结论</a></li><li><a href="report.html">主实验进度</a></li></ul></body></html>'
    (OUT / "adaptation-review.html").write_text(doc)
    print(json.dumps(dict(protocol_checks=protocol_checks, comparisons=c), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
