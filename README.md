# PEVFormer

**Polygon Edge and Vertex-Aware Transformer for Building Contour Extraction from Remote Sensing Images**

[项目仓库](https://github.com/entropia-lw/PEVFormer)

论文正在投稿中，尚未发表，目前不提供论文链接。

PEVFormer 面向遥感图像中的建筑物矢量轮廓提取。模型以建筑物候选框为输入，通过边界感知特征增强（BAFE）和几何感知层级解码（GAHD）预测有序边界采样点及其角点分类，生成建筑物多边形。

> 发布状态：小中建筑物推理代码、匹配权重和两张结果示例已整理。提供的参考预测已完成独立 CPU 指标重算；完整模型前向仍待兼容 GPU 环境验收。权重外部下载链接待补充。

## 1. 发布范围

本仓库提供 CrowdAI 小中建筑物模型的预训练推理与评测，包括：

- 推理所需的网络定义、配置和可变形注意力算子。
- 预训练模型下载说明。
- 检测候选框格式转换。
- 多边形预测与 COCO 格式结果导出。
- COCO Segmentation AP/AR 评估。
- 两张预测结果示例图。

本次发布不包含完整训练流程、大建筑物模型及双分支预测合并。仅使用本仓库的小中建筑物模型，不代表复现包含大建筑物分支的完整系统结果。

## 2. 方法概览

- **BAFE：Boundary-Aware Feature Enhancement。**包含边界引导采样（BGS）、上下文增强边界定位（CBL）和自适应特征融合（AFF），增强建筑物边界相关的 RoI 特征。
- **GAHD：Geometry-Aware Hierarchical Decoding。**结合逐层顶点预测与几何顶点优化（GVR）细化坐标及角点分类。训练监督策略不作为额外推理层运行。

结果示例见第 8 节，两张图均由提供本次权重的作者确认来自该模型。

## 3. 环境准备

### 3.1 参考环境

原云端运行环境为：

| 组件 | 版本或配置 |
|---|---|
| 操作系统 | Ubuntu 18.04 |
| Python | 3.8 |
| PyTorch | 1.9.0 + CUDA 11.1 |
| torchvision | 0.10.0 + CUDA 11.1 |
| Detectron2 | 0.6 |
| GPU | RTX 3090，24 GB |

上述环境为原云端参考环境。整理版完整模型前向仍需在兼容环境中验收；本机已完成 CPU 指标重算及 BAFE/GVR 模块对照，不等同于整网 CUDA 推理通过。

### 3.2 获取代码

```bash
git clone https://github.com/entropia-lw/PEVFormer.git
cd PEVFormer
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

### 3.3 创建环境并安装依赖

```bash
conda create -n pevformer python=3.8
conda activate pevformer

python -m pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
```

Detectron2 需与 PyTorch/CUDA 版本匹配，可参考[官方安装说明](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)。上述 Linux 组合对应的安装示例为：

```bash
python -m pip install detectron2==0.6 \
  -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
```

安装其余推理依赖：

```bash
python -m pip install -r requirements.txt
```

仅重算已有预测的 COCO 指标时，可使用 `requirements-eval.txt`；不需要加载模型或使用 GPU。它不能替代模型推理环境。

### 3.4 编译可变形注意力

从仓库根目录执行，需具备匹配的 CUDA Toolkit 和编译器：

```bash
cd pevformer/ops
sh make.sh
cd ../..
```

检查依赖与 GPU：

```bash
python -c "import torch, detectron2, MultiScaleDeformableAttention; print(torch.__version__); print(torch.cuda.is_available())"
```

推理包名为 `pevformer`。已训练参数的注册名称保留旧名称以兼容原 checkpoint，无需修改权重文件。

## 4. 数据准备

CrowdAI 数据集体积较大，不随本仓库上传至 GitHub。**训练集和验证集的下载、预处理及相关标注准备，均参考 [RoIPoly 仓库的 data/crowdai 目录](https://github.com/HeinzJiao/RoIPoly/tree/main/data/crowdai)下对应的 README 文件。**请按其中训练集和验证集各自的说明操作，并遵守数据集使用条件。

本仓库只提供预训练推理与评估，运行时只需准备验证集；训练集说明用于保持与上游数据处理协议一致，不表示本仓库提供训练流程。

### 4.1 推理与评估所需文件

- 验证集图像。
- 清洗后的真实多边形标注，用于计算 AP/AR。
- Sparse R-CNN（Swin-B）生成的小中建筑物候选框，用于多边形模型推理。

请区分真实标注与检测候选框，不要相互替代，也不要更改图像 ID 或图像文件名。小中建筑物候选框的分流应遵循上游流程；检测框面积与真实多边形面积的处理用途不同。

### 4.2 目录结构

以下目录与本 README 的推理、评估命令对应：

```text
PEVFormer/
├── data/
│   └── crowdai/
│       └── val/
│           ├── images/
│           │   ├── 000000000000.jpg
│           │   └── ...
│           ├── annotation_clean.json
│           └── annotation_sparsercnn_swinb_1930059_sm.json
├── configs/
├── pevformer/
├── scripts/
└── weights/
```

文件用途：

- `annotation_clean.json`：真实多边形标注，传入评估入口。
- `annotation_sparsercnn_swinb_1930059_sm.json`：小中建筑物检测候选框，传入推理入口。
- `images/`：验证集原始图像。

如果按上游说明得到的文件名或存放位置不同，请相应调整下文命令中的路径，保持图像与标注的对应关系。

推理命令中的 `--train-json` 和 `--train-path` 是现有入口的参数名称，此处仍应传入验证数据。

## 5. 预训练模型

小中建筑物模型：

- 文件：`weights/model_2842654.pth`。
- 配置：`configs/pevformer.res50.crowdai.sm.yaml`。
- checkpoint 记录的迭代编号：2842654。
- 文件大小：179,073,053 字节。
- SHA256：`1bdbb56ff1354b61eb3c6cc0556f06d4491d6299e0925a2c6214d630253c019b`。
- 外部下载地址：**待补充**。权重不随 Git 源码提交。

下载后保持下列路径：

```text
weights/
└── model_2842654.pth
```

该权重包含边界先验、置信度门控和可学习温度的参数，BAFE 与 GVR 子模块已通过严格加载检查。推理入口对整个模型执行严格加载；出现参数缺失或形状冲突时会报错，不会静默跳过。

使用配套检测候选框时，无需重新运行检测器。若希望从任意图像生成候选框，还需另行准备 Sparse R-CNN（Swin-B）的配置、权重和推理环境；本次目录未包含该检测器发布包。

## 6. 多边形推理

使用 ResNet-50 小中建筑物配置和提供的预训练权重。推理代码已整理，完整图像前向尚未在本机验收；建议在上述云端参考环境运行。

从仓库根目录执行，确保使用新的输出目录：

```bash
RUN_DIR="outputs/sm_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"

set -o pipefail
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_change.py \
  --config-file configs/pevformer.res50.crowdai.sm.yaml \
  --output "$RUN_DIR" \
  --iter 2842654 \
  --corner-threshold 0.4 \
  --train-json data/crowdai/val/annotation_sparsercnn_swinb_1930059_sm.json \
  --train-path data/crowdai/val/images \
  --opts MODEL.WEIGHTS weights/model_2842654.pth \
  2>&1 | tee "$RUN_DIR/inference.log"
```

该入口中的 `--iter` 用于输出文件命名。示例设为 2842654，因此输出为：

```text
outputs/<本次运行目录>/
├── predictions_2842654.json
└── inference.log
```

预测 JSON 记录图像 ID、类别、置信度、bbox 和多边形坐标。角点阈值为 0.4，保留至少三个顶点且面积不小于 10 的多边形；正式参考结果将记录使用的全部参数。

如修改 `--iter`，请同步修改下一节的预测文件路径。

## 7. COCO Segmentation 评估

使用真实多边形标注和本次生成的预测结果。若新开终端，请先将 `RUN_DIR` 设置为第 6 节实际使用的目录。

```bash
python scripts/evaluation.py \
  --gt-file data/crowdai/val/annotation_clean.json \
  --dt-file "$RUN_DIR/predictions_2842654.json" \
  --eval-type coco_iou \
  --output "$RUN_DIR" \
  2>&1 | tee "$RUN_DIR/evaluation.log"
```

输出指标文件：

```text
test.stats.predictions_2842654.json
```

当前保留原云端评估协议，以便与历史实验比较：

- 采用多边形分割 IoU，输出 AP、AP50、AP75、分尺寸 AP 及 AR 系列结果。
- 原实现只在预测 JSON 出现的图像上评估，不等同于默认覆盖全部验证图像的协议。
- 原 AR 汇总包含 AR、AR50、AR75 和分尺寸 AR，不等同于标准默认 AR@1、AR@10、AR@100 汇总。
- 若输出目录已有同名 stats 文件，原实现会跳过计算。对新预测使用新目录，避免误读旧结果。

如额外提供标准全验证集评估，将单独记录协议与参考结果，不与上述结果混用。

## 8. 结果与可视化

### 8.1 定量结果

下列结果来自本次提供的小中建筑物模型参考预测，并已在本机 CPU 上使用原评估协议独立重算。全部 12 项指标与提供的参考统计逐项一致。

| 指标 | 结果（%） |
|---|---|
| AP | 83.8830 |
| AP50 | 94.5579 |
| AP75 | 91.3786 |
| AR | 87.9980 |

完整指标见 `reference_results/test.stats.predictions_2842654.json`。参考预测共 485,762 个多边形，覆盖 57,177 张图像；原始验证列表有 60,317 张图像。评估图像范围和 AR 汇总定义见第 7 节。

上述数值只对应本次公开的小中建筑物模型，不包含大建筑物分支或双分支合并结果。指标重算不等同于在本机重新执行整网推理；本机验证边界见 `VALIDATION.md`。

### 8.2 预测结果示例

以下展示用户当前提供的两张可视化结果，包含建筑物区域的框、轮廓线与点标记。颜色图例将在作者确认其具体含义后补充。

![CrowdAI 验证图像 245 的可视化结果](visualize/000000000245.jpg)

*示例 1：验证图像 245，展示建筑物区域的框与轮廓点。*

![CrowdAI 验证图像 247 的可视化结果](visualize/000000000247.jpg)

*示例 2：验证图像 247，展示密集排列建筑物的可视化结果。*

本次提供两张示例图、参考指标和可执行评估入口。完整参考预测 JSON 体积较大，外部下载链接待补充；视频和 GIS 文件不作为本次发布的必备材料。

## 9. 使用说明

- 本次只提供小中建筑物模型，不包含大建筑物分支及其合并结果。
- RoIPoly 对 CrowdAI 使用小中/大建筑物分别训练的说明可见其 README；本仓库明确限定本次公开的模型范围。
- 仅进行预训练推理，不需要准备训练集，也不需要生成训练用的固定长度采样标签。
- 自定义图像需先生成候选框，并核对图像 ID、尺寸、类别和框坐标格式。
- 类名与配置采用 PEVFormer、BAFE、BGS、CBL、AFF、GAHD 和 GVR 命名。为兼容 checkpoint，部分内部参数注册路径保留原名称；不要手动批量替换权重键。
- 增强后权重与源码必须一致；出现 missing keys、unexpected keys 或形状冲突时，应先核对版本。
- 已有预测的 AP/AR 计算可单独进行；模型前向仍依赖 Detectron2 和自定义 CUDA 算子。

## 10. 致谢

本项目基于 [RoIPoly](https://github.com/HeinzJiao/RoIPoly) 开展，并受益于 [Sparse R-CNN](https://github.com/PeizeSun/SparseR-CNN)、[Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) 和 [RoomFormer](https://github.com/ywyue/RoomFormer) 的公开实现。感谢相关作者的贡献。

## 11. 许可证

**以下为许可证草案，正式许可将以发布时确认的根目录 LICENSE 文件及适用的第三方许可为准。**

对于作者有权授权的项目代码，拟采用与 RoIPoly 一致的 [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International（CC BY-NC-SA 4.0）](https://creativecommons.org/licenses/by-nc-sa/4.0/) 许可方向：

- 使用、修改或再分发时，应保留适当署名、许可链接及修改说明。
- 不得将受该许可约束的材料用于商业目的。
- 再分发改编材料时，应遵守相同方式共享要求。
- 软件按现状提供，不作保证。

本项目所包含的第三方组件继续遵守各自许可证，保留原作者的版权和许可声明；本节不替代或覆盖第三方授权条件。预训练权重和数据集的使用条件将在相应下载说明中明确。

## 12. 问题反馈

请通过 [GitHub Issues](https://github.com/entropia-lw/PEVFormer/issues) 反馈问题，并提供环境版本、执行命令和错误日志。
