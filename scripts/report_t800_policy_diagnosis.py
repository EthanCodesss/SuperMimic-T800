"""Render the frozen-checkpoint diagnostic delivery from saved measurements."""
import hashlib
import html
import json
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/t800-diagnosis-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/t800_policy_diagnosis'
MODES=[('rl_lab_4000','旧 RL Lab 尺度','RL Lab scale'),('torque_kp_4000','新 torque/Kp 尺度','Torque/Kp scale')]


def load(path): return json.loads(path.read_text())
def mean(v): return float(np.mean(v))
def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def main():
    delivery=dict(status='completed_frozen_checkpoint_diagnosis',training_resumed=False,optimizer_steps=0,
        scope='Two 4000-update checkpoints from one training seed; 50 full-clip physical episodes and fixed-start model/physical/parameter diagnostics.',runs={})
    physical_rows=[];robust_rows=[];gradient_rows=[];goal_rows=[];parameter_rows=[];consistency_rows=[];links=[]
    fig,axes=plt.subplots(1,3,figsize=(15,4.2))
    for num,(mode,label,english) in enumerate(MODES):
        folder=OUT/mode
        proto=load(folder/'protocol.json')
        physical=load(folder/'physical-results.json')
        status=load(folder/'physical-status.json')
        objective=load(folder/'objective-results.json')
        goal=load(folder/'goal-response.json')
        parameter=load(folder/'parameter-results.json')
        assert status['status']==objective['status']==parameter['status']=='passed'
        assert proto['checkpoint_sha256']==objective['checkpoint_sha256']==parameter['checkpoint_sha256']
        base=next(x for x in physical if x['case']=='baseline' and x['seed']==0)
        repeat=next(x for x in physical if x['case']=='baseline_repeat')
        assert abs(base['full_clip_normalized_return']-repeat['full_clip_normalized_return'])<1e-5
        summary=dict(checkpoint=proto['checkpoint'],checkpoint_sha256=proto['checkpoint_sha256'],physical_status=status,
            physical=[],robustness=[],directional={},parameter_probe=parameter['parameter_probe'],reference=parameter['reference'],
            goal_response=goal,goal_encoder=objective['goal_encoder'],validation=objective['validation'])
        for name in ['baseline','both_0.1','both_0.25','both_0.5','both_1']:
            x=next(x for x in physical if x['case']==name and x['seed']==0)
            summary['physical'].append(x)
            physical_rows.append([label,f"{x['alpha']:g}",f"{x['full_clip_normalized_return']:.4f}",f"{x['joint_rmse_rad']:.4f}",f"{x['elbow_rmse_rad']:.4f}",'完成' if x['completed_clip'] else '失败'])
        for name in ['baseline','both_0.25','both_0.5']:
            xs=[x for x in physical if x['case']==name and x['reset_noise']]
            x=dict(case=name,episodes=len(xs),completed=sum(t['completed_clip'] for t in xs),
                **{k:mean([t[k] for t in xs]) for k in ['full_clip_normalized_return','joint_rmse_rad','elbow_rmse_rad']})
            summary['robustness'].append(x)
            robust_rows.append([label,name,f"{x['full_clip_normalized_return']:.4f}",f"{x['joint_rmse_rad']:.4f}",f"{x['elbow_rmse_rad']:.4f}",f"{x['completed']}/{x['episodes']}"])
        for H in [1,8,32]:
            ms={x['beta']:x for x in objective['model'] if x['horizon']==H}
            ps={x['beta']:x for x in objective['physical'] if x['horizon']==H}
            values={}
            for k in ['reward_return','critic_tail','objective']:
                grad=np.array(ms[0]['derivatives'][k]['beta'])
                wm_fd=(np.array(ms[.05][k])-np.array(ms[-.05][k]))/.1
                physical_fd=(np.array(ps[.05][k])-np.array(ps[-.05][k]))/.1
                values[k]=dict(model_autograd_mean=mean(grad),model_finite_difference_mean=mean(wm_fd),
                    physical_finite_difference_mean=mean(physical_fd),model_physical_sign_agreement=mean(np.sign(grad)==np.sign(physical_fd)),
                    autograd_vs_model_fd_relative_l2=float(np.linalg.norm(grad-wm_fd)/max(np.linalg.norm(grad),1e-12)))
            summary['directional'][str(H)]=values
        d=summary['directional']['32']
        cancellation=-d['critic_tail']['model_autograd_mean']/d['reward_return']['model_autograd_mean']
        summary['critic_cancellation_fraction']=cancellation
        summary['noise_probe_means']=[{k:mean(v['beta']) for k,v in x['derivatives'].items()} for x in objective['noise_probe']]
        gradient_rows.append([label,f"{d['reward_return']['physical_finite_difference_mean']:+.4f}",f"{d['reward_return']['model_autograd_mean']:+.4f}",f"{d['critic_tail']['model_autograd_mean']:+.4f}",f"{d['objective']['model_autograd_mean']:+.4f}",f"{100*cancellation:.1f}%"])
        cp={x['beta']:x for x in parameter['continuations']}
        summary['physical_continuation_derivatives']={k:mean((np.array(cp[.05][k])-np.array(cp[-.05][k]))/.1) for k in ['reward_return','first_32_reward_return','remainder_reward_return']}
        comps=parameter['parameter_probe']['components']
        for k,c in comps.items():
            predicted=-c['raw_gradient_elbow_mse_improvement_dot']/c['gradient_norm']
            measured=c['target_response']['all']['elbow_target_mse_derivative']
            assert abs(predicted-measured)<max(1e-5,.01*abs(predicted)),(mode,k,predicted,measured)
            response=c['target_response']['all']['mean_elbow_target_derivative']
            parameter_rows.append([label,k,f"{c['cosine_with_elbow_target_mse_improvement']:+.4f}",f"{measured:+.4f}",f"{response[0]:+.4f}",f"{response[1]:+.4f}"])
        goal_rows.append([label,f"{100*objective['goal_encoder']['abs_latent_above_099_fraction']:.1f}%",
            f"{mean(np.abs(goal['joints']['16']['autograd_per_state'])):.5f}",f"{mean(np.abs(goal['joints']['21']['autograd_per_state'])):.5f}",
            f"{goal['ablations']['goal_10_frames_later']['elbow_target_change_rms_rad']:.4f}"])
        r=parameter['reference'];v=objective['validation']['cpu_replay_vs_gpu']['32']
        consistency_rows.append([label,f"{goal['history_max_abs_error']:.2e}",f"{goal['goal_max_abs_error']:.2e}",f"{v['joint_position_rmse_rad']:.2e}",f"{r['body_position_rmse_m']:.2e}",f"{r['reference_pose_analytic_normalized_reward']:.7f}"])
        curve=[next(x for x in physical if x['case']==name and x['seed']==0) for name in ['both_-0.1','baseline','both_0.1','both_0.25','both_0.5','both_1']]
        xs=[x['alpha'] for x in curve]
        axes[0].plot(xs,[x['full_clip_normalized_return'] for x in curve],marker='o',label=english)
        axes[1].plot(xs,[x['elbow_rmse_rad'] for x in curve],marker='o',label=english)
        keys=['reward_return','critic_tail','objective']
        axes[2].bar(np.arange(3)+(num-.5)*.35,[d[k]['model_autograd_mean'] for k in keys],width=.35,label=english)
        delivery['runs'][mode]=summary
        links.append(f'<li>{html.escape(label)}：<a href="{mode}/protocol.json">协议与哈希</a> · <a href="{mode}/physical-results.json">物理回放</a> · <a href="{mode}/objective-results.json">模型与 critic</a> · <a href="{mode}/parameter-results.json">参数梯度与参考一致性</a> · <a href="{mode}/goal-response.json">目标响应</a></li>')
    axes[0].set(xlabel='Elbow correction fraction',ylabel='Full-clip normalized return',title='Real simulation: reward')
    axes[1].set(xlabel='Elbow correction fraction',ylabel='Elbow RMSE (rad)',title='Real simulation: elbow tracking')
    axes[2].set(xticks=np.arange(3),xticklabels=['Reward','Critic tail','Total'],ylabel='d objective / d correction',title='32-step model objective')
    for ax in axes:
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    axes[2].axhline(0,color='black',lw=.8)
    fig.tight_layout()
    fig.savefig(OUT/'diagnosis.png',dpi=180)
    fig.savefig(OUT/'diagnosis.svg')
    plt.close(fig)
    delivery['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'scripts/diagnose_t800_policy_commands.py',ROOT/'scripts/diagnose_t800_objective.py',ROOT/'scripts/diagnose_t800_policy_gradient.py']}
    (OUT/'delivery.json').write_text(json.dumps(delivery,ensure_ascii=False,indent=2)+'\n')
    body='''<h1>T800 短动作策略误差诊断</h1><p>2026-10-28 · 冻结 4000 次检查点 · 训练保持停止 · 优化器更新 0 次</p>
<div class="callout"><b>结论：</b>当前证据把主要排查方向指向第 7 部分的奖励／critic 取舍，以及第 4 部分共享策略的梯度分配。未发现参考动作、观测排列或 PD 映射的接线错误。世界模型能够识别本次肘部修正的收益；但这不意味着所有多步梯度都准确。尚不能宣称找到了一个已被训练对照验证的唯一根因。</div>
<p>此次使用同一短动作、两套动作尺度各一个 4000 次检查点。总计完成 50 次完整物理回放。两套检查点都来自同一个训练随机种子；五组初始扰动是回放鲁棒性检查，不是五次独立训练。</p>
<h2>1. 真实仿真：策略确实漏掉了可实现的局部改进</h2>
<p>只将左右肘 PD 目标按比例 β 拉向当前参考角，前 10 个控制步平滑开启，其余动作仍由原策略根据真实历史产生。β=0 为原策略，β=1 为肘部完全采用参考角。这是输出干预，不是重新训练或一个已经验证的部署方案。</p>'''
    body+=table(['尺度','β','完整回报','全关节 RMSE/rad','两肘 RMSE/rad','完成'],physical_rows)
    body+='''<p>新尺度 β=0.5 时，全关节误差降低约 9.6%，两肘误差降低约 44.8%，回报提高约 2.5%。这比固定根部 PD 测试更直接地表明：当前自由运动中，较准确的肘部姿态是可实现的。旧尺度大幅修正时根部跟踪等分项变差，整体回报下降；并非主要由 action-rate 惩罚单独造成。</p>
<p>β=1 的新尺度两肘误差进一步降到约 0.121 rad，但总回报约 0.502，与原策略接近。当前优化目标没有直接的关节角 RMSE 项；提高关节准确度不一定提高所有奖励分项。</p>
<h3>五组配对初始扰动的均值</h3>'''
    body+=table(['尺度','干预','回报','全关节 RMSE/rad','两肘 RMSE/rad','完成'],robust_rows)
    body+='<img src="diagnosis.png" alt="Physical intervention and gradient comparison">'
    body+='''<h2>2. 世界模型知道改善方向，critic 会抵消部分收益</h2>
<p>从真实基线轨迹的第 20、40、60、80、100、120、140、160 帧分别起步。使用上游 imagine 和 policy_objective 原函数，比较 1、8、32 步；物理有限差分 β=±0.05，模型另计算自动微分。32 步时，两套尺度的纯奖励改善方向在八个起点均与物理回放一致。下表为八个起点平均 dJ/dβ，单位不是完整动作 normalized return。</p>'''
    body+=table(['尺度','物理奖励导数','模型奖励导数','模型 critic 尾项','合计','critic 抵消比例'],gradient_rows)
    body+='''<p>旧尺度中 critic 几乎抵消平均奖励导数；新尺度约抵消四分之一，合计仍为正。加入三组观测噪声后，新尺度合计方向均为正，旧尺度出现正负变化。不能据此断言 critic 全错：它要估计后续平衡代价，且大幅干预的真实收益确实可能变差。</p>
<p>补充检查将修正只开启 32 步，然后恢复原策略至片段结束。新尺度实际后续收益导数平均约 +0.018，而模型 critic 尾项约 −0.022；旧尺度实际后续约 −0.022，而模型 critic 尾项约 −0.087。说明尾项存在方向／幅度偏差的迹象，但有限片段回报与训练中的 bootstrap 尾项口径不同，不能把它当作完整 critic 标定。</p>
<h2>3. 梯度传到网络后，关节改善被明显稀释</h2>
<p>计算同八个起点的完整策略参数梯度，并在固定真实观测上做对称、无状态参数扰动，测量输出的变化方向。没有修改网络参数，没有执行 Adam。下表中的 MSE 导数越负越有利；目标角导数为沿各分量单位参数梯度的响应，不能视为真实下一次 Adam 更新量。当前两肘目标整体偏向伸直，因此平均负方向通常更接近参考。</p>'''
    body+=table(['尺度','梯度分量','与肘部 MSE 改善方向余弦','肘部目标 MSE 导数','左肘目标角导数','右肘目标角导数'],parameter_rows)
    body+='''<p>新尺度奖励梯度与肘部目标改善方向的余弦约 +0.096，加入 critic 后降到 +0.038；整体仍可改善，但平均左肘目标反而往远离参考的方向移动。旧尺度合计余弦约 −0.029，局部参数梯度倾向于增大肘部目标误差。这是目标之间及共享参数之间存在竞争的直接证据，不是网络断梯度的证据。</p>
<h2>4. 观测和目标没有接错，但局部目标响应弱</h2>
<p>固定真实本体观测，只在未来十帧中同时给一个肘关节参考位置加小偏移，速度等其他特征保持不变。下表为该关节 PD 目标对参考角的导数绝对值平均，避免正负抵消。整体目标换成后续十帧时策略仍明显响应，不能说策略忽略了全部目标。</p>'''
    body+=table(['尺度','|latent|>0.99 的比例','左肘 |d目标/d参考|','右肘 |d目标/d参考|','后移目标后两肘变化 RMS/rad'],goal_rows)
    body+='''<p>新尺度编码器约 78% 的输出处于 tanh 饱和区，旧尺度约 49%。这可能削弱精细参考角变化的学习，但单个动作可以通过少量相位特征编码，因此饱和比例本身不能证明表示崩溃。需要编码器／动作残差形式的训练对照才能判断因果。</p>
<h2>5. 接线与数值一致性检查</h2>'''
    body+=table(['尺度','历史观测最大误差','目标特征最大误差','32 步 CPU/GPU 关节 RMSE','参考 FK 位置 RMSE','理想参考奖励'],consistency_rows)
    body+='''<p>参考 FK 的位置、姿态和速度误差均为约 10⁻⁷ 量级。理想参考状态的奖励约 1.0，说明没有发现参考数组、刚体姿态或奖励坐标系自相矛盾。它仍不证明整个动作在接触动力学下可精确实现。</p>
<p>物理梯度比较采用完整积分后状态的解析奖励，以匹配世界模型的奖励时刻；完整动作回放保留环境原本的奖励计算。环境刚体派生量与积分状态存在一个物理子步的时序区别，不能混用两类数值。无干预 imagine 包装与原函数的差异不超过原函数自身重复运行的数值差异，奖励最大差小于 10⁻⁵。</p>
<h2>6. 按原八个部分定位</h2>'''
    body+=table(['部分','此次证据与判断'],[
        ['1. 参考动作','运动学和奖励一致性通过；完整动力学可行性仍未被全面证明。'],
        ['2. 机器人与环境','自由运动肘部干预可改善；没有证据表明 PD 不能完成当前肘部姿态。'],
        ['3. 观测和目标','历史排列、帧对齐一致；单关节目标响应弱，整体目标输入有效。'],
        ['4. 策略网络','梯度可达；共享参数产生左右肘不同的改善方向，编码器有饱和信号。'],
        ['5. 动作映射与 PD','物理映射正确；改变尺度仍会改变优化条件和奖励取舍，不能从 PD 正常推出尺度无影响。'],
        ['6. 回放与世界模型','本次方向的短期及 32 步奖励预测有效；critic 敏感的多步状态误差仍需保留。'],
        ['7. 想象策略优化','当前最值得优先验证：奖励/critic 抵消、共享策略梯度与关节角准确度对齐较弱。'],
        ['8. 评估和记录','基线重复回放复现；所有结果来自冻结检查点，不是新训练收益。']])
    body+='''<h2>7. 接下来的最小验证</h2>
<p>继续保持训练停止。本次检查尚不足以宣布某项修改可以将回报提高到 0.9。若恢复短动作训练，优先做同一个新尺度检查点、相同种子及更新预算的三组小对照：原目标续训；只降低 actor 中 critic 尾项权重；保留原 critic 并加入小权重关节角辅助奖励。分别回答“尾项是否过强”和“奖励对姿态准确度是否给足信号”，不要同时修改观测、尺度和网络。</p>
<p>若这两项仍无法改善，再单独比较参考姿态加残差的动作形式，或缓解编码器饱和。任何辅助奖励或残差映射都属于偏离上游默认配置的实验，需明确记录；这次没有实施这些训练改动。评估继续同时看原版 normalized return、关节 RMSE、左右肘 PD 目标误差及扰动完成率。</p>
<h2>原始结果与限制</h2><ul>'''+''.join(links)+'''</ul>
<p><a href="delivery.json">结构化交付数据</a> · <a href="diagnosis.svg">矢量图</a></p>
<p>限制：一条短动作、一个训练种子、每个模型八个诊断起点；输出干预和参数导数不是长期训练消融。实际训练还有随机起点、域随机化、探索噪声、Adam 动量等，本次没有重建完整训练批次。结论只对当前检查点和被测方向成立。</p>'''
    doc='<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>T800 策略误差诊断</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#25313d;background:#fafbfd}h1,h2,h3{color:#102e47}h2{margin-top:36px}table{border-collapse:collapse;width:100%;font-size:14px;margin:18px 0;background:white}th,td{border:1px solid #d8e0e8;padding:9px;text-align:left}th{background:#eaf1f7}img{width:100%}.callout{padding:20px;background:#e8f3ef;border-left:5px solid #237859}a{color:#176591}</style><body>'+body+'</body></html>'
    (OUT/'report.html').write_text(doc)
    print(OUT/'report.html')


if __name__=='__main__': main()
