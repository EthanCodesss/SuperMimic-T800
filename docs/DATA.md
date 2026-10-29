# 数据复现

本实验使用 Ubisoft LAFAN1 的完整 `walk1_subject1.bvh`（7840 帧，30 Hz）。请从[官方项目](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)获取，并遵守其数据许可证；文件放在 `data/raw/lafan1/walk1_subject1.bvh`。本仓库不重新分发原始或重定向动作。

实验下载包 SHA256：`ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`。原始 BVH SHA256、完整 IK 配置和任务残差保存在 [reference-provenance.json](results/reference-provenance.json)。

按照根目录 README 的两步流程先生成完整默认参考，再采用 [reference-ik-seed.json](results/reference-ik-seed.json) 重定向。该初始姿态在训练前通过十个固定候选、前 600 个源帧的 GMR 任务残差选出；它只影响初始 IK 状态，没有逐帧重启、修改目标或平滑。

完成后输出 13066 帧、50 Hz 的 NPZ。每帧包含 25 个关节位置/速度、身体世界位置/四元数/线速度/角速度；单位米、弧度，z 向上，四元数 wxyz。MuJoCo 自由根角速度与数据世界角速度的变换由现有转换/状态代码处理。

[reference-array-hashes.json](results/reference-array-hashes.json) 保存原实验各数组的 shape、dtype 与内容 SHA256。`scripts/verify_release_reference.py` 对数组逐项验证，不依赖 ZIP 封装时间戳或来源字符串。若有不一致，请保留新文件并检查软件版本、初始姿态、原文件和求解器，不能把新的参考当作原实验完全相同的输入。

训练时参考在首尾边界的处理、未来帧采样和重置沿用固定上游流程。当前最佳结果使用完整动作；早期 199 帧短片段的结果属于历史诊断，不能混入完整动作的对比。
