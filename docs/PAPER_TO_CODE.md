# 论文—代码可追踪矩阵

来源：用户提供的 `manuscript.pdf`，标题 *Dynamic Semantic Adaptation for Efficient Large Language Model Inference*，SHA-256 `E3D83F0162A0624D16CFC5EBFB769CB11107F14DBC53E4078CF5CE76F4D34553`。PDF 共 17 页。页码以 PDF 页序为准。

| 论文内容 | PDF 页 | 代码 | 验证 |
|---|---:|---|---|
| 语义相似、锚点缓存、残差映射、漂移与一次纠正，式(1)-(6) | 4–5 | `anchor.py` | `test_anchor.py` |
| 语义熵、信息增益、依赖跨度、复杂度融合，式(7)-(14) | 5–6 | `signals.py`, `pruning.py` | `test_signals.py`, `test_pruning.py` |
| 三段剪枝策略、重要性与确定性 top-k，式(15)-(17) | 6 | `pruning.py` | `test_pruning.py` |
| 置信度、不确定性、精度敏感度与四条精度路径，式(18)-(27) | 6–8 | `precision.py` | `test_precision.py` |
| 局部刷新、端到端误差和阈值反馈 | 8–9 | `feedback.py`, `controller.py` | `test_feedback.py`, `test_controller.py` |
| 统一运行时顺序（Algorithm 1） | 8–9 | `controller.py`, `adapters/llama.py` | `test_controller.py`, `test_llama_adapter.py` |
| Appendix B 辅助训练目标 | 14–15 | `calibration.py`, `training.py` | `test_calibration.py`, `test_training.py` |
| Appendix C judge prompt | 15 | `judge.py` | `test_judge.py` |
| A100 环境、batch/context、warm-up/measure | 9–10 | `paper_a100.yaml`, `environment-a100.yml`, `experiments.py` | `test_config.py`, `test_provenance.py` |
| 四个数据集和样本数 | 9–10 | `data/manifests/`, `data.py` | `test_data.py` |
| 主表、消融、gate、mask、质量表 | 10–13 | `paper_reference/*.csv`, `reporting.py` | `test_reporting.py` |

## 实现解释

1. `AnchorBank` 只让前一步完整计算得到的层状态成为下一步候选锚点；当前步 staging 在 `finish_step` 后才 release，避免把近似输出当作新完整锚点。
2. `RuntimeController.enter_layer` 固定执行：共同回退检查 → 锚点近似 → 剪枝/刷新 → 精度选择。消融开关不改变其余机制。
3. `DSALlamaAdapter` 在近似层仍合成并驻留 K/V，使下一 token 的缓存长度正确；完整层使用实际 LLaMA decoder layer。
4. INT8/INT4 是对隐藏激活的确定性参考 fake-quant。论文未公开的 resident weight-pack kernel 由 `ResidentPackStore` 接口表达，但不伪装成已有融合实现。
5. 校准保存冻结完整模型的监督；训练优化器只持有 `AuxiliaryModules.parameters()`。

## 无法从 PDF 唯一确定的内容

- 原作者源代码、提交版本和 checkpoint；
- 512 条 ShareGPT 的精确 ID、数据版本、去重和切分；
- 权重打包布局、量化校准细节、融合 kernel 与编译 flags；
- Vicuna-80 私有/人工参考答案；
- 历史 judge 服务的后端快照与潜在系统更新。

这些项目均在输出/README 中显式标记，不以猜测补齐。
