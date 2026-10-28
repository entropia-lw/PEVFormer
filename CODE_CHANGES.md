# 代码修改对照

原项目 D:\PEVFormer_github 保持不变。权重、参考结果和两张图片均使用用户本次提供的文件，不重写其内容。旧检查记录保留在 _release_audit，当前结果见 enhanced_20261028。

| 变更类别 | 原实现 | 发布实现 |
|---|---|---|
| 模型入口 | 包含训练与多种损失 | 仅保留构建和推理前向 |
| 数据读取 | 训练、推理共用 | 保留推理图像与候选框处理 |
| 模块名称 | 内部工程名称 | 采用论文缩写 |
| 已训练参数路径 | 原 checkpoint 路径 | 原样保留 |
| 权重加载 | 宽松框架加载 | 整网严格加载 |
| 配置 | 训练配置与旧命名 | 推理配置和论文命名 |
| 中间特征 | 首次推理自动写文件 | 移除调试文件写入 |
| 评估 | 多指标外部依赖 | 仅保留原 COCO 路由 |
| 指标定义与后处理 | 原算法 | 不变 |

## 源码与发布文件

- 原 `roipoly/detector.py` → `pevformer/model.py`：保留骨干构建、标准化、候选框 padding、特征选择和前向输出；移除损失、训练目标、监控和调试输出。
- 原 `roipoly/contour_enhancer.py` → `pevformer/bafe.py`：保留推理分支，移除训练 warmup/监控分支。
- 原 `roipoly/geometric_refine.py` → `pevformer/gvr.py`：保留几何编码、融合、自注意力和残差输出，移除中间特征保存。
- 原 `roipoly/head.py` → `pevformer/head.py`：保留初始化和推理运算，移除训练输出/监控及调试文件写入。
- 原 `roipoly/deformable_transformer.py` → `pevformer/deformable_transformer.py`：前向运算不变，必要工具改为包内导入。
- 原 `roipoly/dataset_mapper.py` → `pevformer/dataset_mapper.py`：保留原推理时的图像读取、空变换、框转换和过滤；移除训练标签与增强依赖。
- 原 `roipoly/config.py` → `pevformer/config.py`：修改配置命名，移除不使用的 Swin/TTA/优化器默认注册。
- 原 `util/misc.py` 中的逆 sigmoid → `pevformer/utils.py`；新增兼容原 PyTorch 1.9 与本机新 PyTorch 的 checkpoint 读取。
- 原 `roipoly/ops` → `pevformer/ops`：仅保留编译和运行所需源码，不复制历史构建产物。
- 原 `scripts/evaluate_change.py` → 同名发布入口：新包和配置名；严格加载原参数键；移除自动关闭模型模块的行为，使用明确配置；保留角点阈值、面积过滤、置信度和 COCO 输出计算。
- `scripts/evaluation.py`：仅保留原 coco_iou 路由；`scripts/eval_coco.py` 的指标计算保持不变。
- `configs/Base-PEVFormer.yaml`、`configs/pevformer.res50.crowdai.sm.yaml`：保持有效推理参数，移除训练调度项，默认加载用户提供的新权重。

## 名称对应

- `RoIPoly` → `PEVFormer`。
- `RoIPolyDatasetMapper` → `PEVFormerDatasetMapper`。
- `add_roipoly_config` → `add_pevformer_config`。
- `ContourFeatureEnhancer` → `BAFE`。
- `SparseFeatureExtractor` → `BGS`。
- `DenseFeatureExtractor` → `CBL`。
- `FeatureFusion` → `AFF`。
- `DynamicHead` → `PolygonHead`。
- `TransformerHead` → `GAHD`（推理部分）。
- `GeometricVertexRefinement` → `GVR`。
- 配置命名空间改为 `MODEL.PEVFormer`；模块开关与配置节点使用 `USE_BAFE`、`BAFE`、`USE_GVR`、`GVR`。

已训练注册名 `head`、`transformer_head`、`contour_enhancer`、`sparse_feature_extractor`、`dense_feature_extractor`、`feature_fusion` 和 `geo_refine` 原样保留。这样无需篡改或转换 checkpoint，避免改键引入加载差错。

## 未纳入发布

大建筑物模型、训练入口、损失实现、历史版本、Swin 骨干/TTA、训练日志、训练数据处理、训练环境、编译二进制和中间特征文件未纳入此推理包。数据、大权重和完整参考预测在 .gitignore 中排除；原文件未删除。第三方声明移至 THIRD_PARTY_NOTICES.md 与 third_party。

完整整网前向暂因环境未验证，不能以模块对照代替整网验收。当前无需用户再次确认即可完成的整理和测试已执行；剩余 GPU 验收可用 README 命令在原云端环境运行。
