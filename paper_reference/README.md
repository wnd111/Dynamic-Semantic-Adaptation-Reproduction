# Paper reference tables

这些 CSV 是从用户提供 PDF 的表格人工转录的 `paper_reference`，用于绘图目标线和差异对照，不是本代码实测结果。

- `table2_stochastic_latency.csv`：论文随机/自适应机制延迟结果。
- `table7_ablation.csv`：完整 controller 与单组件移除。
- `table8_gate_sweep.csv`：gate 阈值、近似率、延迟、ROUGE-L、judge。
- `table9_mask_comparison.csv`：不同 mask 策略。
- `table10_quality.csv`：论文质量指标汇总。

报告器在存在实测 `gate_sweep.csv` 时优先使用实测文件；否则使用这里的参考表，并在 `summary.json` 中记录来源。禁止删除来源字段后把本目录内容宣称为本机实测。
