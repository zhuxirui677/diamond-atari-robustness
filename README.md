# 世界模型测试代码与结果

本仓库用于共享世界模型实验。当前内容统一放在 **[DIAMOND/](DIAMOND/)** 文件夹中。
后续其他模型可使用独立文件夹，避免混淆不同模型和实验协议。

## DIAMOND 文件夹里面有什么

|路径|内容|
|---|---|
|[DIAMOND/README.md](DIAMOND/README.md)|详细任务定义、扰动说明、运行方式、步数口径与结果限制|
|[DIAMOND/atari_protocol.py](DIAMOND/atari_protocol.py)|Breakout、Pong、Boxing任务阈值、H=100及统计指标|
|[DIAMOND/diamond_atari_runner.py](DIAMOND/diamond_atari_runner.py)|原协议：动作丢弃与加性高斯噪声的ALE／DIAMOND配对测试|
|[DIAMOND/diamond_libero_plus_runner.py](DIAMOND/diamond_libero_plus_runner.py)|LIBERO-plus Sensor Noise接入DIAMOND的测试入口|
|[DIAMOND/libero_plus_image_functions.py](DIAMOND/libero_plus_image_functions.py)|固定上游版本的五类图像扰动函数|
|[DIAMOND/libero_plus_provenance.json](DIAMOND/libero_plus_provenance.json)|扰动源码版本及来源记录|
|[DIAMOND/DIAMOND-Atari-Tasks.ipynb](DIAMOND/DIAMOND-Atari-Tasks.ipynb)|原协议的VS Code Notebook控制器|
|[DIAMOND/DIAMOND-LIBERO-Plus.ipynb](DIAMOND/DIAMOND-LIBERO-Plus.ipynb)|高斯模糊冒烟测试Notebook|
|[DIAMOND/DIAMOND-LIBERO-Plus-All.ipynb](DIAMOND/DIAMOND-LIBERO-Plus-All.ipynb)|五类Sensor Noise的完整测试控制器；代码准备好不等于结果已完成|
|[DIAMOND/results/](DIAMOND/results/)|实测逐条记录、汇总CSV、manifest、PASS、案例数据和示例GIF|
|[DIAMOND/reports/](DIAMOND/reports/)|任务表、细分表，以及高斯模糊实测结果与扰动参数Excel|
|DIAMOND/test_*.py|任务协议及图像扰动验证|
|DIAMOND/requirements-*.txt|Notebook控制器与图像扰动的额外依赖|

## 当前已完成的测试

|实验|实际状态|
|---|---|
|原协议：动作丢弃＋加性高斯噪声|Breakout/Pong/Boxing，共1440条rollout完成；每条件每环境20个案例|
|LIBERO-plus高斯模糊冒烟|Breakout，σ=1和10像素，加normal/sham，共16条rollout完成；每条件每环境2个案例|
|其余Sensor Noise全档位完整测试|代码已准备，尚无已核实的完成结果|
|光照、额外初始状态扰动|尚未接入Atari适配|
|与ARAT实验设置对齐|待共享对方具体协议|

两批已完成实验的结果分目录保存；冒烟结果不能当作完整实验结论。
所有任务测试预算H=100决策步，frame skip=4；模型历史条件窗口为4步。

## 如何查看与运行

先阅读 [DIAMOND详细说明](DIAMOND/README.md)。
运行Notebook或单元测试时，工作目录应为 `DIAMOND/`。
Notebook已清除输出，服务器、用户名和Slurm节点使用占位符，需要改成自己的配置。

## 如何提交修改

其他人可以fork仓库、修改后提交Pull Request，详见 [CONTRIBUTING.md](CONTRIBUTING.md)。
直接上传到本仓库分支需要所有者授予collaborator权限；公开可访问不等于所有人都能直接push。
