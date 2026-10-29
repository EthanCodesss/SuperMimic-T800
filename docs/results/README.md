# 已完成实验的证据

这里复制了原实验的 JSON，不改写数值、源码哈希或机器路径；`manifest.json` 标明原始位置与 SHA256。机器路径是历史来源记录，不是安装要求。原始完整运行日志、训练检查点和逐帧参考数据留在实验机器；Release 另附最终检查点及可视化。

[发布验证](publication-validation.json) 另记录了 7 项测试、参考数组重建校验，以及在独立目录迁移检查点后完成的整段 GPU 评估。此验证复用已安装的软件环境和校验过的 vendor 快照，没有重新安装整套依赖或重新训练。

| 文件 | 内容 |
|---|---|
| `anchor-head-final.json` | IEEE FP32 完整动作物理评估，8104 更新 |
| `anchor-head-benchmark.json` | 上游 benchmark 流程的 T800 适配结果，默认 TF32 |
| `anchor-head-robustness.json` | 同一个模型，五个重置扰动种子 |
| `anchor-head-config.json` / `anchor-head-execution.json` | 训练配方与实际执行路径 |
| `anchor-head-protocol.json` / `anchor-head-audit.json` | 不可变实验协议与组合适配检查 |
| `baseline-final.json` / `baseline-benchmark.json` | 同预算无 anchor、头部策略控制基线 |
| `baseline-joint-errors.json` | 基线各关节误差，含头部/身体拆分 |
| `reference-*` / `data-protocol.json` | 参考动作、初始 IK 姿态与质量检查 |

两组训练都修复了本地 FK 编译问题。组合组的全局误差大幅降低，身体 23 关节 RMSE 则从 0.0665 增至 0.0850 rad。头部误差显著改善，不能因此省略身体误差的变化。

组合组各阶段：2000 更新的 IEEE 回放在 135.16 秒因根高度终止，完整片段归一化回报 0.3353；同一检查点 TF32 benchmark 完成。4052 和 8104 更新在两种评估路径均完成，IEEE return 分别为 0.9277 和 0.9556。提前终止的运行不能仅以执行段平均奖励描述为完整动作成功。

研究限制：一条训练动作、一个训练种子、两项干预同时改变；重置扰动不替代多种子训练、未见动作测试或真机验证。没有根据本实验声称优于不同本体/数据集的上游 G1 或 SONIC。
