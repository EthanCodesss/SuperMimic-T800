"""User-requested 4000-update comparison, kept separate from the final study."""
import html
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"reports/t800_action_scale_study/interim_4000"
SOURCE="https://github.com/CuboiLeo/SuperMimic-Reproduced/blob/ba6105349069bf040f1fdad60873d13138d505a2/docs/EXPERIMENTS.md"


def read(path):
    return json.loads(path.read_text())


def table(headers,rows):
    row=lambda x,tag:"<tr>"+"".join(f"<{tag}>{html.escape(str(v))}</{tag}>" for v in x)+"</tr>"
    return "<table>"+row(headers,"th")+"".join(row(r,"td") for r in rows)+"</table>"


def main():
    stopped=(OUT.parent/"user-stop-record.json").exists()
    a,b=[read(OUT/f"{name}.json") for name in ("control","treatment")]
    older=read(ROOT/"reports/training/t800_short_torque_kp_2000/final.json")
    ar=[r for r in read(ROOT/"reports/training/t800_short_noanchor_8000/robustness.json") if r["checkpoint_updates"]==4000]
    br=[read(OUT/f"treatment_reset_seed_{i}.json") for i in range(1,6)]
    assert len(ar)==len(br)==5
    terms=b["mean_step_reward_terms"]
    ceiling=b["normalized_return"]+(.02-terms["motion_global_root_pos"]-terms["motion_global_root_ori"])/.1
    reduction=1-b["joint_position_rmse_rad"]/a["joint_position_rmse_rad"]
    result=dict(status="interim_4000_complete_training_stopped_by_user" if stopped else "interim_4000_complete_8000_training_continues",control=a,treatment=b,
                treatment_2000=older,joint_rmse_reduction_fraction=reduction,
                treatment_root_terms_perfect_other_terms_fixed_return=ceiling,
                counterfactual_scope="Arithmetic substitution of the two root reward terms only; not a bound on what retraining or anchor observations could change in other terms",
                control_robustness=ar,treatment_robustness=br,
                upstream_reference=dict(source=SOURCE,e1c=dict(anchor=False,updates=8104,eval_return=.676),
                    e3=dict(anchor=True,clips=40,environment_iterations=48000,approx_policy_updates=19200,
                            training_return=.79,eval_clip_mean=.904,eval_step_weighted=.908)),
                pause_and_evaluation=read(OUT/"pause-and-evaluation.json"))
    trajectory=np.load(OUT/"treatment.trajectory.npz")
    actions=np.load(OUT/"treatment.actions.npz")
    result["elbow_command_diagnosis"]={b["action_parameterization"]["joint_names"][j]:dict(
        reference_mean_rad=float(trajectory["reference_qpos"][:,7+j].mean()),
        PD_target_mean_rad=float((-.25+actions["raw_actions"][:,j]*actions["scales"][j]).mean()),
        actual_mean_rad=float(trajectory["qpos"][:,7+j].mean())) for j in (16,21)}
    (OUT/"comparison.json").write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n")
    rows=[]
    for key,label,mult in (("normalized_return","Normalized return",1),("joint_position_rmse_rad","关节 RMSE (rad)",1),
                           ("mean_root_position_error_m","根位置平均误差 (cm)",100),("mean_abs_yaw_error_deg","平均 yaw 误差 (°)",1)):
        rows.append([label,f"{a[key]*mult:.4f}",f"{b[key]*mult:.4f}"])
    fractions=[]
    for key in a["reward_fraction_of_term_max"]:
        fractions.append([key,f"{100*a['reward_fraction_of_term_max'][key]:.2f}%",f"{100*b['reward_fraction_of_term_max'][key]:.2f}%"])
    robust=[]
    for label,records in (("原固定尺度",ar),("新计算尺度",br)):
        robust.append([label,f"{sum(r['completed_clip'] for r in records)}/5",
                       f"{np.mean([r['joint_position_rmse_rad'] for r in records]):.4f}",
                       f"{np.mean([r['normalized_full_clip_return'] for r in records]):.4f}"])
    page="<!doctype html><html lang='zh-CN'><meta charset='utf-8'><title>T800 4000 次中期评估</title><style>body{font:16px/1.7 system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#182b3a}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ccd7df;padding:8px;text-align:left}th{background:#edf4f8}</style><body><h1>T800 动作尺度：4000 次中期评估</h1>"
    progress="训练随后按用户要求停止，8000 次验证未完成。" if stopped else "8000 次同预算验证仍在运行。"
    page+=f"<p>两组均为 4000 次策略更新、同一短动作、无 anchor 观测。关节 RMSE 改善 {100*reduction:.1f}%，标准回放回报基本相同。新组从 2000 到 4000 次回报提高，但关节 RMSE 未继续改善；目前不支持单靠动作尺度就能解决主要局部姿态误差。{progress}</p>"
    page+=table(["指标","原固定尺度","新计算尺度"],rows)
    page+="<h2>五组初始扰动</h2>"+table(["组别","完成数","平均关节 RMSE","平均完整回报"],robust)
    page+="<h2>各项奖励 / 各自上限</h2>"+table(["项","原固定尺度","新计算尺度"],fractions)
    page+=f"<p>保持其他项不变，只把新组根位置、根朝向奖励替换为满分，回报为 {ceiling:.4f}。这说明当前损失主要不在这两个根部项；不能将此代数替换当成对重新训练收益的上界。</p>"
    page+="<h2>与上游数值的关系</h2><p>本表约 0.503 是 evaluation normalized return，当前训练日志 train_ret 约 0.43。单回合归一化为 sum(reward)/(0.1 × 实际执行步数)。额外的完整片段回报将未执行部分记零；本次均完整完成，两者相等。</p>"
    page+=f"<p>上游 e1c：G1 单动作、无 anchor，8104 次更新，完整动作评估 0.676。e3：G1、40 条 LAFAN1、anchor 观测，48000 次环境迭代（约 19200 次更新），训练 0.79、评估动作均值 0.904、按步数加权 0.908。它在约 6400/9600 次更新时已达 0.792/0.824；不能把我们当前差距全归因于更新次数。机器人、数据与观测不同，仍不是同任务对照。<a href='{SOURCE}'>上游实验记录</a>。</p>"
    page+="<h2>评估记录</h2><p>为响应用户即时评估请求，训练进程暂时 SIGSTOP，评估后 SIGCONT 原地恢复；没有重新载入训练检查点、清空回放或重置 RNG。标准基线与新组均本次重测，五组扰动的基线复用既有 4000 次记录。GPU 重复回放存在微小数值差异；五组扰动不是五个训练种子。</p><p><a href='comparison.json'>完整结果</a> · <a href='pause-and-evaluation.json'>暂停与恢复记录</a> · <a href='../reward-frame-audit.json'>Anchor 身体离线评分审计</a></p></body></html>"
    (OUT/"report.html").write_text(page)
    print(OUT/"report.html")


if __name__=="__main__":
    main()
