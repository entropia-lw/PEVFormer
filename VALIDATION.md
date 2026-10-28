# 验证报告：增强后小中建筑物发布包

| 检查 | 结果 |
|---|---|
| Python 3.8 语法、注释与文档字符串扫描 | 18 个 Python 文件通过 |
| YAML 解析与权重路径 | 通过 |
| BAFE 原/改名后严格加载 | 37 项全部匹配 |
| GVR 原/改名后严格加载 | 40 项全部匹配 |
| BAFE/GVR CPU 前向对照 | 各 2 个张量用例，逐元素一致 |
| 预测头位置编码对照 | 逐元素一致 |
| 评估 CLI | 通过 |
| 真实数据参考预测 AP/AR 重算 | 全部 12 项完全一致 |
| 本机整网前向 | 未通过环境导入 |
| 整网原/整理版预测一致性 | 尚未验证 |
| CUDA 算子重新编译与运行 | 尚未验证 |

## 实测指标

AP=0.8388301346990471，AR=0.8799800282633958。全部指标与用户提供的参考 stats 文件完全相同，最大绝对差 0。重算使用真实 annotation_clean.json 和完整参考预测 JSON，保留原预测图像范围与 AR50/AR75 汇总协议。

当前预测有 485,762 个多边形，覆盖 57,177 张图像；验证列表 60,317 张。参考预测的评估不等同于重新生成预测。

## 模块对照边界

使用 checkpoint 的实际张量、原实现与整理实现的真实 torch 运算，不使用 mock。BAFE 使用随机和零 RoI 特征；GVR 使用随机与退化几何坐标，最大误差均为 0。模块测试不等于整图测试。

原源码触发 torch.meshgrid 的未来接口提醒；为保持原 PyTorch 1.9 行为，本次未更改该计算。未执行依赖升级。

## 本机整网限制

执行发布版真实推理命令，在导入 Detectron2 时失败。将原项目携带的框架源码加入搜索路径再次探测，也因 detectron2._C 编译扩展缺失而失败。自定义 MultiScaleDeformableAttention 同样未在本机获得兼容运行时。未使用 mock、替代算子或随机参数使流程表面通过。

本机是 Python 3.12 / PyTorch 2.7.1+cu118 / RTX 5060 Laptop（sm_120）；原云端是 Python 3.8 / PyTorch 1.9.0+cu111 / RTX 3090。未安装依赖、创建环境或更改已有环境。完整模型严格加载与 CUDA 前向需在兼容云端验证。

## 详细证据

- `_release_audit/enhanced_20261028/validation.json`：检查结果、失败命令和错误输出。
- `_release_audit/enhanced_20261028/full_metrics/test.stats.predictions_2842654.json`：独立重算指标。
- `_release_audit/enhanced_20261028/generation_manifest.json`：文件与命名映射。
- `_release_audit/enhanced_20261028/reference_metadata.json`：参考结果规模。

两张可视化图已实际查看。整理期间原 PNG 被外部替换为图像 245/247 的 JPG，README 已引用当前文件并采用中性图注；颜色语义尚未确认。最初的模型来源确认不用于推断替换图片的具体颜色含义。
