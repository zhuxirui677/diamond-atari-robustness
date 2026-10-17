# DIAMOND Atari robustness experiments

DIAMOND 的 Atari 正常／扰动对比代码与实测结果。此仓库使用预训练 DIAMOND，不是机械臂 LIBERO 实验。

## 已完成与尚未完成

|实验|状态|范围|
|---|---|---|
|原协议动作丢弃＋加性高斯噪声|已完成，Slurm job 29|Breakout/Pong/Boxing；20案例；12条件；ALE/WM；1440 rollout|
|LIBERO-plus 高斯模糊冒烟|已完成，Slurm job 32|Breakout；2案例；normal/sham/σ=1/σ=10；ALE/WM；16 rollout|
|五类 Sensor Noise 全档位完整测试|代码已准备，未确认运行完成|motion blur / Gaussian blur / zoom blur / fog / glass blur，各10档|
|Light Conditions / 额外 Initial State 扰动|尚未接入|需共享 Atari 适配方法与参数|
|与 ARAT 的实验对齐|待对方协议|不能将现有参数宣称为已完成跨模型对齐|

`results/` 保存逐条轨迹记录、汇总、manifest、PASS与示例GIF；`reports/` 保存Excel。
报告分别对应各自实验批次；早期TASKS表中的窗口字段应按下述统一口径解释。

## 任务与步数

- H=100 **决策步**，frame skip=4；提前终止不重置续跑。
- Breakout：至少1个正奖励步；Pong：至少得1分；Boxing：至少5个正奖励步，不等于5拳。
- 阈值是项目短任务定义，不代表完整游戏通关。
- 扩散模型历史条件窗口4步；官方代码训练采样序列6个时间点（4历史＋2预测训练）。
- 官方actor-critic训练想象horizon=15，burn-in=4；不要与本项目测试H=100混用。
- 单条训练轨迹长度未核实，H/训练轨迹长度仍未知。

## 扰动协议

旧协议：动作以p=0.05/0.10/0.20/0.30/0.40替换为NOOP；加性高斯噪声σ=0.01/0.02/0.05/0.10/0.20，单位为RGB[0,1]。

LIBERO-plus：复用固定版本原始图像函数。高斯模糊severity=1…10对应σ=1…10 **像素**，不是加性噪声。
其他原版参数见 `libero_plus_image_functions.py`。通过 `DIAMOND_PLUS_FAMILY` 选择类型。
所有图像扰动作用于预处理后的64×64 RGB策略输入，世界模型历史保持干净；两域接收实际执行动作。
各条件匹配ALE状态、RNG、观测/动作历史及策略隐状态，normal/sham核对轨迹哈希。
原版 stochastic 函数的NumPy RNG被隔离，避免污染策略随机数。

## 运行

1. 本机安装 `requirements-controller.txt`，在VS Code打开Notebook。
2. 将Notebook里的 `YOUR_NETID`、`YOUR_CLUSTER_HOST`、`YOUR_SLURM_NODE` 和NAS/环境路径改为自己的配置；GPU分配应遵守所在集群规则。
3. 主机指纹文件 `~/.ssh/known_hosts_gix` 必须预先通过可信渠道核验。密码仅在人手操作的getpass框输入，不保存到代码或Notebook输出。
4. `DIAMOND-LIBERO-Plus.ipynb` 只提交高斯模糊冒烟；`DIAMOND-LIBERO-Plus-All.ipynb` 会先冒烟，再对通过的五类依次提交完整作业。
5. GPU作业在官方DIAMOND仓库目录运行本仓库runner；需要原DIAMOND依赖、ROM及checkpoint下载访问权限。这里不包含ROM或模型权重。
6. motion blur需要系统ImageMagick及Wand；缺少时测试明确失败，不以替代算法冒充原版。

本地协议测试：`python -m unittest test_atari_protocol.py test_libero_plus_sensor.py`（后者需要图像依赖）。
实际结果必须以 `PASS`、manifest及完整逐条记录为准，不能只凭Slurm退出队列判断成功。

## 结果边界

高斯模糊冒烟仅N=2，不支持稳健排名。ALE和WM分别闭环运行，动作可能分叉；域间差距不是同动作预测误差，也不是校准后的预警信号。
GIF为干净游戏画面，不能代表策略实际看到的扰动画面。

## 来源与贡献

- [DIAMOND](https://github.com/eloialonso/diamond)，checkpoint revision `5d4abca9af6ab1b3ab1c6fc228cacb7281130f7f`。
- [LIBERO-plus](https://github.com/sylvestf/LIBERO-plus)，原版图像函数来源revision `4976dc30028e805ff8094b55501d532c48fec182`；详见provenance。
- 上游代码及资产遵循各自许可；本仓库未授予超出上游许可的使用权。
- 其他人可fork仓库，在分支修改后提交Pull Request。直接push需要仓库所有者单独授予collaborator权限。
