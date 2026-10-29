# 固定版本与数据来源

> 2026-10-30 发布说明：下文的短动作协议为历史诊断。当前推荐的完整动作入口、8104 更新结果与复现步骤见根目录 [README](../README.md)。`bootstrap.sh` 使用 `requirements-linux-cu128-portable.lock.txt`，仅将原始环境锁中本机路径形式的 packaging 条目改为相同版本 `packaging==26.3`；原始环境锁保持不动。

- `t800-single-motion-policy-ieee-baseline.json`：本轮 T800 基线的不可变协议，记录源码/数据哈希、
  本体差异、头部和跟踪刚体选择、2,000 次更新预算及评估门槛。
  `scripts/prepare_t800_baseline.py` 检查原始默认配置仅存在本体维度差异，再生成该协议。
  `scripts/run_training_stage.py` 启动前验证文件哈希，预算结束后自动评估并停止。
  `t800-single-motion-baseline.json` 保留为先前 TF32 诊断运行的协议；该轮因补充数值对照未通过而停止。

- `sources.lock.json`：上游提交与源码文件 SHA256；未修改算法快照在 `vendor/supermimic_upstream`。
- `upstream-dependencies.json`：从 CuboiLeo 原始 `uv.lock` 提取的版本清单。
- `conda-linux-64.lock.txt`：Python 3.12.16 等 Conda 基础包的明确下载地址。
- `requirements-linux-cu128.lock.txt`：当前兼容环境的 Python 依赖版本。
- `compatibility-constraints.txt`：必须维持的 Torch / Warp / MuJoCo 核心版本。
- `build-constraints.txt`：构建 mjlab 和 SuperMimic 时使用的后端版本。
- 各实验的 JSON：由 `scripts/run_experiment.py` 写入的实际训练参数。
- `strict-source-lock.json`：固定严格复现入口所允许的源码；可选诊断补丁未应用。
  `scripts/train_strict.py` 默认只输出配置；`--mode probe` 检查原始规模容量，
  `--mode train` 运行原始训练循环，运行配置与结果保存在 `reports/strict/`。
  此处旧的短训练 JSON 为历史诊断配方，不能作为上游默认配置。
- `scripts/train_memory.py` 使用独立的激活重计算实现，保留原始训练配置；
  `reports/microbatch/<run-name>/execution.json` 记录该实现、依赖和源码哈希，
  `config.json` 仅与原始配方相差运行名称或恢复路径。
  显存优化检查点有明确执行方式标记，不能混作未修改执行路径的严格复现检查点。

当前兼容环境使用 Torch 2.11.0+cu128、Warp 1.18.0 的 CUDA 12 wheel、MuJoCo / MuJoCo Warp 3.16.0。
CuboiLeo 原始锁定的是 Torch 2.14.1+cu130，不能直接照搬到本机 570.211.01 驱动。
因此 mjlab 固定提交的 `pyproject.toml` 仅放宽了 Torch 最低版本；这是本地兼容配置，不是原始软件栈的逐版本复现。
执行 `bash scripts/bootstrap.sh` 可在名为 `supermimic` 的环境中安装相同版本。
普通 Python 包默认使用阿里云镜像，可通过 `SUPERMIMIC_PYPI_INDEX` 覆盖，不修改全局 pip 配置。
`scripts/fetch_sources.py` 可恢复缺失的源码快照，并校验固定提交的文件哈希；
`--verify-only` 只校验现有文件，不下载或覆盖修改。

模型与控制参数说明见 `assets/robots/t800/SOURCE.json`。
T800 的关节定义和 MJCF 来自众擎 GMR，已与同一时期的 native SDK XML 交叉核对；
PD、力矩、armature、默认姿态及动作缩放来自固定版本的众擎 RL Lab tracking 配置。
SDK 舞蹈部署配置和训练配置不能混用数组下标，数据必须按关节名称对齐。

数据来源：

- G1 LAFAN1 CSV：[lvhaidong/LAFAN1_Retargeting_Dataset](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset)，30 Hz、根四元数 xyzw。
- 原始 LAFAN1：[Ubisoft La Forge](https://github.com/ubisoft/ubisoft-laforge-animation-dataset)，下载包 SHA256 `ea918082b500a5d158e9d3aa39039df04cd42e25f5c02fe8f7e88e8e9365a977`。
- T800 舞蹈 CSV：[engineai_rl_lab](https://github.com/engineai-robotics/engineai_rl_lab/blob/c53f7ec614e34f1ea514cefb935894082f3d4c2a/datasets/tracking/t800/dance_t800.csv)，30 Hz、SDK 关节顺序、xyzw。
- G1 公开检查点：[CuboiLeo e8-boneseed-100k](https://github.com/CuboiLeo/SuperMimic-Reproduced/releases/tag/e8-boneseed-100k)。它使用 BONES-SEED 训练，本次仅用少量公开 LAFAN1 动作验证运行。

所有参考动作统一转换为 50 Hz、米、弧度、z 向上。NPZ 四元数使用 wxyz，根角速度为世界坐标；MuJoCo 状态的根角速度为本体坐标。
LAFAN1 及其重定向数据保持本地使用，不作为本项目代码许可证下的数据重新分发；原始许可证为 CC BY-NC-ND 4.0。
没有下载需要账户授权的 BONES-SEED 原始数据，也没有下载 SMPL-X 人体模型。
