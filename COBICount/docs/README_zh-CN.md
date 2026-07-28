# COBICount 中文使用说明

本仓库是论文 **COBICount: Candidate-Origin Bias Isolation for Single-Source Generalizable Remote Sensing Object Counting in Unseen Scenarios** 的整理版研究代码。

## 本次整理解决的问题

原始训练脚本会在训练过程中构建 DOTA Proxy 数据加载器、周期性计算目标域指标并保存目标域最优检查点。即使该检查点未用于正式结果，这种实现也容易与论文声明的严格 `source-only` 协议产生冲突。

整理后的代码将流程完全分离：

- `cobicount-train` 只导入并构建 RSOC 源域数据；
- `cobicount-evaluate` 只在训练完成、检查点冻结后加载 DOTA；
- 训练过程只保存 `best_source_val.pth`，不再生成任何 `best_dota_proxy` 检查点；
- 默认训练轮数由 200 改为论文所写的 80；
- 删除所有本地 Windows 绝对路径，全部改为命令行参数；
- 增加 Python 包结构、测试、CI、许可证、引用文件和 GitHub 模板。

## 安装

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e .
```

在 CUDA 机器上，建议先按照本机 CUDA 版本安装对应的 PyTorch，再执行 `pip install -e .`。

## 训练

```bash
cobicount-train ^
  --rsoc-root F:\path\to\RSOC_building\building ^
  --output-dir runs\cobicount_rsoc_building ^
  --epochs 80 ^
  --batch-size 6 ^
  --device cuda
```

Windows PowerShell 也可以写成一行。训练命令不会导入 DOTA 数据集。

## DOTA 训练后测试

```bash
cobicount-evaluate ^
  --checkpoint runs\cobicount_rsoc_building\best_source_val.pth ^
  --dataset dota-full ^
  --dota-image-root F:\path\to\DOTA\images ^
  --dota-annotation-root F:\path\to\DOTA\annotations ^
  --class-filter "small vehicle" ^
  --file-list F:\path\to\test_small_vehicle.txt ^
  --output-dir outputs\dota_sv
```

## 需要注意

1. 默认 `--val-split test_data` 是为了兼容原 RSOC 文件夹结构。若后续重新划分源域训练/验证集，应将独立验证集目录名称传给 `--val-split`。
2. DOTA Proxy 是只保留含目标裁剪的补充诊断协议，不能与完整图像 DOTA 指标直接比较。
3. Audit 四个通道只是辅助诊断响应，不是经过校准的车辆、建筑、船舶和背景分类器。
4. 连续预测数量来自得分图求和，诊断点数量来自峰值提取，两者不要求相等。
5. 不要把 `.pth`、数据集、运行输出和本地路径提交到 Git 历史中。
