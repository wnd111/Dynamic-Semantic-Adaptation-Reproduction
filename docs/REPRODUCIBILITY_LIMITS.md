# 复现限制与结果解释

## 可完整复现

- 论文描述的决策逻辑、门限、最大近似链、漂移纠正和回退顺序；
- 复杂度/置信度辅助网络、校准 trace、伪标签和冻结基础模型的训练流程；
- 确定性数据选择、任务指标、评审请求缓存、五种消融/敏感性入口；
- A100 环境检查、同步延迟协议、manifest、SHA-256、测试、报告和发布归档。

## 不能承诺精确复现

1. **作者材料缺失**：PDF 没有源码和 checkpoint，clean-room 初始化与训练不可逐位相同。
2. **ShareGPT 子集不唯一**：只给出 512 条数量，缺少样本 ID/版本；任何擅自抽取都会产生不可验证差异。本包不隐藏这一点。
3. **性能 kernel 缺失**：PyTorch eager 的参考 fake-quant 不等价于作者可能使用的融合 INT4/INT8 kernel 和 resident pack；因此绝对延迟只反映本 backend。
4. **评审模型时变**：论文 judge ID 是历史预览模型。当前兼容模型的分数是新实验，必须标为 `current_noncomparable`。
5. **Vicuna-80 答案缺失**：公共题集不含统一标准答案，本包不会生成伪参考。

## 正确报告方式

发布结果时至少同时提供：配置文件、`run_manifest.json`、GPU 型号、包版本、git commit、数据记录 SHA-256、checkpoint 元数据、每个指标的样本数和 `result_kind`。比较论文表格时，应将论文值标为“reported/reference”，将本机值标为“measured”。

只有在获得作者的源码、权重、数据索引和 kernel 后，才应使用“exact reproduction”描述；当前包应称为“complete clean-room reproduction package”。
