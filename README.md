# Document Denoising with LightGBM

基于多尺度图像特征与 LightGBM 像素回归的文档图像去噪项目。模型从受污染的灰度扫描图中恢复文字与纸张背景，输出每个像素在 `[0, 1]` 范围内的灰度预测。

## 项目来源

本项目是以下练习与竞赛的个人实现，与主办方无隶属关系：

- [AI Coding Gym — Denoising Dirty Documents](https://aicodinggym.com/challenges/mle/denoising-dirty-documents)
- [Kaggle — Denoising Dirty Documents](https://www.kaggle.com/competitions/denoising-dirty-documents)

题目与数据集来自上述来源；本仓库仅提供实现代码与说明，不分发原始数据。请从来源页面获取数据，并遵守其使用条款。

## 方法

1. 提取像素邻域、不同尺度的高斯平滑、中值、局部最小值和最大值等特征。
2. 利用局部最大值、平滑及形态学闭运算估计纸张背景，构建亮度差值和比值特征。
3. 从训练图中随机抽样像素，用对应干净图的灰度训练 LightGBM 回归模型。
4. 按图像划分训练与验证集，根据验证 RMSE 选择迭代次数，再用全部有标签图像重新训练。
5. 对测试图逐像素预测，裁剪到 `[0, 1]`，生成比赛要求的 `id,value` 文件。

这是一种基于手工图像特征的监督学习方案，不使用预训练模型。

## 已有结果

| 评估 | RMSE（越低越好） |
| --- | ---: |
| 验证样本直接使用脏图灰度 | 0.157925 |
| LightGBM 本地验证，预测裁剪后 | 0.015319 |
| AI Coding Gym 提交评分 | 0.01416 |

记录日期：2026-09-12。平台结果来自 AI Coding Gym，不能视为 Kaggle 官方排行榜成绩。

本次数据包含 115 张有标签图和 29 张测试图。验证按图像尺寸分层，划分为 95 张训练图、20 张验证图；每张训练图抽取 8,000 个像素，每张验证图抽取 25,000 个像素。上表本地 RMSE 基于验证像素样本，不是验证图的全像素评估。最终拟合使用全部 115 张有标签图，每图抽取 8,000 个像素。随机种子为 `20260912`，最终选择 1,599 轮。

验证尚未按相同干净底稿分组，也未进行多次划分或交叉验证，因此本地结果的稳定性与对全新底稿的泛化能力仍需进一步评估。库版本和平台差异也可能影响复现结果。

## 使用

建议使用 Python 3.10 或更高版本，并在项目虚拟环境中安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows 可用 `.venv\Scripts\activate` 激活环境。macOS 上 LightGBM 还需要可被动态链接器找到的 OpenMP 运行库 `libomp.dylib`；它不包含在本仓库中。

从题目来源获取数据，解压后按以下目录组织：

```text
document-denoising-lightgbm/
├── solve.py
├── requirements.txt
└── data/
    ├── train/
    ├── train_cleaned/
    ├── test/
    └── sampleSubmission.csv
```

`train/` 与 `train_cleaned/` 中相同文件名应对应同一张图。运行：

```bash
python solve.py
```

输出为 `outputs/predictions.csv`，列为 `id,value`。ID 使用 `图号_行_列`，行列编号从 1 开始；程序按图片文件名字典序、图内行优先顺序写出。已提交版本的 5,789,880 个像素 ID 与本次数据的示例文件逐行核对一致，预测值均在 `[0, 1]` 范围内。程序本身不自动提交，也不自动执行该逐行校验；换用数据后应重新检查示例格式。

脚本会重新训练模型并覆盖同名预测文件，模型权重暂不单独保存。

## 仓库范围

仓库仅保留源代码、依赖清单、说明和忽略规则。数据集、预测 CSV、模型权重、虚拟环境、缓存、编译产物、日志及本机工具配置均不纳入版本管理。发布使用独立的初始提交，不携带原工作区的历史记录。

## 后续改进

- 检查重复底稿并按底稿分组验证，补充全像素 RMSE 和多次划分。
- 分析文字边缘、文字内部与背景区域的误差。
- 比较采样策略、背景估计窗口及 LightGBM 参数。
- 尝试小型卷积网络，并在可靠验证集上评估模型融合。

## 数据致谢

依据题目介绍，数据集由 RM.J. Castro-Bleda、S. España-Boquera、J. Pastor-Pellicer 和 F. Zamora-Martinez 创建，UCI Machine Learning Repository 提供过数据托管。研究或发表时请查阅来源页面的引用要求，包括：

Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository. Irvine, CA: University of California, School of Information and Computer Science.
