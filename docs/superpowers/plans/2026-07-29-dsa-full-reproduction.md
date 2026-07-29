# Dynamic Semantic Adaptation Full Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a tested, self-contained A100 reproduction package for the controller, calibration/training workflow, four benchmark evaluations, ablations, statistics, and reports described in the supplied DSA manuscript.

**Architecture:** A pure-PyTorch controller implements the manuscript equations and state machine independently of model integration. A Hugging Face LLaMA adapter, dataset/evaluation layer, and experiment CLI consume that stable controller. Every run writes resolved configuration and provenance; paper-reference, measured, and synthetic results are typed separately.

**Tech Stack:** Python 3.10, PyTorch 2.1.0 + CUDA 12.1, Transformers 4.36.0, Datasets 2.16.x, NumPy, SciPy, pandas, PyYAML, Matplotlib, Typer, pytest, ruff.

## Global Constraints

- Target hardware is one NVIDIA A100 80GB and the paper protocol uses batch size one.
- Base model is `meta-llama/Llama-2-7b-hf`; base-model parameters stay frozen.
- Credentials are read only from `HF_TOKEN` and `OPENAI_API_KEY`.
- No model weights, raw datasets, credentials, caches, or virtual environments enter the archive.
- Default paper constants must match Tables 3, 5, and 6 and Algorithm 1.
- Measured, paper-reference, and synthetic outputs must never be silently mixed.
- A paper-exact judge profile and a current-compatible judge profile must be reported as non-comparable.

---

### Task 1: Package skeleton, configuration, and provenance

**Files:**
- Create: `pyproject.toml`
- Create: `environment-a100.yml`
- Create: `requirements-dev.txt`
- Create: `src/dsa_repro/__init__.py`
- Create: `src/dsa_repro/config.py`
- Create: `src/dsa_repro/provenance.py`
- Create: `configs/paper_a100.yaml`
- Create: `configs/current_compatible.yaml`
- Test: `tests/test_config.py`
- Test: `tests/test_provenance.py`

**Interfaces:**
- Produces: `load_config(path: Path) -> ExperimentConfig`, `write_manifest(run_dir: Path, config: ExperimentConfig, result_kind: str) -> Path`, and immutable nested configuration dataclasses.

- [ ] **Step 1: Write failing configuration and provenance tests**

```python
def test_paper_defaults_match_table_3():
    cfg = load_config(Path("configs/paper_a100.yaml"))
    assert cfg.model.model_id == "meta-llama/Llama-2-7b-hf"
    assert cfg.anchor.gate_threshold == 0.85
    assert cfg.anchor.drift_threshold == 0.15
    assert cfg.anchor.max_chain == 3
    assert cfg.pruning.high_threshold == 0.7
    assert cfg.pruning.low_threshold == 0.3
    assert cfg.feedback.local_interval == 20
    assert cfg.feedback.e2e_interval == 100

def test_manifest_rejects_unknown_result_kind(tmp_path):
    paper_config = Path("configs/paper_a100.yaml")
    with pytest.raises(ValueError, match="result_kind"):
        write_manifest(tmp_path, load_config(paper_config), "unknown")
```

- [ ] **Step 2: Run tests and verify they fail because the package does not exist**

Run: `python -m pytest tests/test_config.py tests/test_provenance.py -q`

- [ ] **Step 3: Implement typed configuration, validation, YAML defaults, and atomic manifests**

```python
@dataclass(frozen=True)
class AnchorConfig:
    gate_threshold: float = 0.85
    drift_threshold: float = 0.15
    max_chain: int = 3
    approximation_error: float = 0.05

def write_manifest(run_dir: Path, config: ExperimentConfig, result_kind: str) -> Path:
    if result_kind not in {"measured", "paper_reference", "synthetic_smoke"}:
        raise ValueError("result_kind must be measured, paper_reference, or synthetic_smoke")
    payload = collect_environment(config, result_kind)
    return atomic_write_json(run_dir / "run_manifest.json", payload)
```

- [ ] **Step 4: Run tests and static imports**

Run: `python -m pytest tests/test_config.py tests/test_provenance.py -q`

Run: `python -m compileall -q src`

### Task 2: Complexity signals and metrics-grade math

**Files:**
- Create: `src/dsa_repro/signals.py`
- Create: `src/dsa_repro/math_utils.py`
- Test: `tests/test_signals.py`

**Interfaces:**
- Produces: `semantic_entropy(logits)`, `information_gain(prefix_probs, next_probs)`, `dependency_span(attention)`, `hidden_ambiguity(current, previous)`, `normalized_l2_error(candidate, reference)`, and `ppr_divergence(controller_logits, full_logits)`.

- [ ] **Step 1: Write equation-level failing tests**

```python
def test_semantic_entropy_is_normalized():
    uniform = torch.zeros(1, 4)
    certain = torch.tensor([[40.0, -40.0, -40.0, -40.0]])
    assert semantic_entropy(uniform).item() == pytest.approx(1.0)
    assert semantic_entropy(certain).item() == pytest.approx(0.0, abs=1e-5)

def test_information_gain_is_symmetric_and_bounded():
    p = torch.tensor([[0.75, 0.25]])
    q = torch.tensor([[0.25, 0.75]])
    assert information_gain(p, q).item() == pytest.approx(information_gain(q, p).item())
    assert 0.0 <= information_gain(p, q).item() <= 1.0

def test_ambiguity_first_position_is_zero():
    h = torch.randn(2, 3, 4)
    a = hidden_ambiguity(h)
    assert torch.equal(a[:, 0], torch.zeros(2))
```

- [ ] **Step 2: Run the signal tests and observe missing-function failures**

Run: `python -m pytest tests/test_signals.py -q`

- [ ] **Step 3: Implement equations 7-9, 16, 21, 27, and normalized ratios with stable clamping**

```python
def semantic_entropy(logits: Tensor, eps: float = 1e-12) -> Tensor:
    probs = logits.float().softmax(dim=-1)
    entropy = -(probs * probs.clamp_min(eps).log()).sum(dim=-1)
    return entropy / math.log(logits.shape[-1])

def information_gain(p: Tensor, q: Tensor, eps: float = 1e-12) -> Tensor:
    p, q = p.float(), q.float()
    m = 0.5 * (p + q)
    js = 0.5 * ((p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1)
                + (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(-1))
    return js / math.log(2.0)
```

- [ ] **Step 4: Run signal tests**

Run: `python -m pytest tests/test_signals.py -q`

### Task 3: Cached-anchor approximation and correction

**Files:**
- Create: `src/dsa_repro/anchor.py`
- Test: `tests/test_anchor.py`

**Interfaces:**
- Produces: `AnchorBank.release_after_step()`, `AnchorBank.nearest_eligible(layer_idx)`, `AnchorApproximator.propose(...) -> ApproximationProposal`, and `AnchorApproximator.validate(...) -> ApproximationDecision`.

- [ ] **Step 1: Write failing tests for anchor timing, similarity, chain cap, correction, and fallback**

```python
def test_anchor_is_not_visible_until_step_release():
    bank = AnchorBank(num_layers=4)
    bank.stage(1, torch.ones(1, 1, 8), kv=None)
    assert bank.nearest_eligible(2) is None
    bank.release_after_step()
    assert bank.nearest_eligible(2).layer_idx == 1

def test_failed_correction_forces_full_layer(make_approximator):
    module = make_approximator(drift_score=0.2, corrected_drift_score=0.2)
    decision = module.validate(torch.zeros(1, 1, 8), torch.ones(1, 1, 8))
    assert decision.use_full_layer
    assert decision.reason == "post_correction_drift"
```

- [ ] **Step 2: Run anchor tests and verify expected failures**

Run: `python -m pytest tests/test_anchor.py -q`

- [ ] **Step 3: Implement equations 1-6 and the Lmax=3 containment rule**

```python
@dataclass(frozen=True)
class ApproximationDecision:
    accepted: bool
    hidden_state: Tensor
    use_full_layer: bool
    corrected: bool
    similarity: float
    drift: float
    reason: str

def normalized_similarity(z_cur: Tensor, z_anchor: Tensor, delta: float = 1e-6) -> Tensor:
    numerator = (z_cur * z_anchor).sum(dim=-1)
    denominator = z_cur.norm(dim=-1) * z_anchor.norm(dim=-1) + delta
    return numerator / denominator
```

- [ ] **Step 4: Run anchor tests**

Run: `python -m pytest tests/test_anchor.py -q`

### Task 4: Predictive attention path pruning

**Files:**
- Create: `src/dsa_repro/pruning.py`
- Test: `tests/test_pruning.py`

**Interfaces:**
- Produces: `ComplexityPredictor.forward(hidden: Tensor, signals: ComplexitySignals) -> Tensor`, `MaskSelector.select(score: Tensor, importance: Tensor, seq_len: int) -> MaskDecision`, `MaskDecision.full(seq_len: int) -> MaskDecision`, and `ProgressiveRefresh.update(step: int, audit_error: float | None, decision: MaskDecision) -> MaskDecision`.

- [ ] **Step 1: Write failing tests for all three mask paths and deterministic routing**

```python
@pytest.mark.parametrize((score, mode, keep), [(0.8, "full", 10), (0.5, "window", 9), (0.2, "topk", 5)])
def test_three_tier_mask(score, mode, keep):
    importance = torch.arange(10, dtype=torch.float32)
    decision = MaskSelector(0.7, 0.3, topk_fraction=0.5, window_fraction=0.9).select(
        torch.tensor(score), importance, seq_len=10
    )
    assert decision.mode == mode
    assert decision.indices.numel() == keep

def test_topk_has_no_rng_dependence():
    importance = torch.tensor([0.4, 0.1, 0.3, 0.2])
    a = MaskSelector.default().select(torch.tensor(0.1), importance, 4)
    b = MaskSelector.default().select(torch.tensor(0.1), importance, 4)
    assert torch.equal(a.indices, b.indices)
```

- [ ] **Step 2: Run pruning tests and observe missing-module failures**

Run: `python -m pytest tests/test_pruning.py -q`

- [ ] **Step 3: Implement equations 10-15, causal inclusion, and periodic/PPR refresh state**

```python
class MaskSelector:
    def select(self, score: Tensor, importance: Tensor, seq_len: int) -> MaskDecision:
        if score.item() > self.high_threshold:
            indices = torch.arange(seq_len, device=importance.device)
            mode = "full"
        elif score.item() >= self.low_threshold:
            start = max(0, seq_len - math.ceil(seq_len * self.window_fraction))
            indices = torch.arange(start, seq_len, device=importance.device)
            mode = "window"
        else:
            count = max(1, math.ceil(seq_len * self.topk_fraction))
            indices = importance[:seq_len].topk(count, sorted=True).indices.sort().values
            mode = "topk"
        return MaskDecision(mode=mode, indices=indices)
```

- [ ] **Step 4: Run pruning tests**

Run: `python -m pytest tests/test_pruning.py -q`

### Task 5: Adaptive precision, local audits, and threshold feedback

**Files:**
- Create: `src/dsa_repro/precision.py`
- Create: `src/dsa_repro/feedback.py`
- Test: `tests/test_precision.py`
- Test: `tests/test_feedback.py`

**Interfaces:**
- Produces: `PrecisionScheduler.score(...)`, `PrecisionScheduler.select(...) -> PrecisionPath`, `fake_quantize(...) -> QuantizedTensor`, `LocalAuditController.observe(...)`, and `EndToEndFeedback.observe(...)`.

- [ ] **Step 1: Write failing tests for ordered thresholds, quantization, and feedback boundaries**

```python
@pytest.mark.parametrize((score, expected), [(0.8, "fp32_acc"), (0.6, "fp16"), (0.4, "int8"), (0.2, "int4")])
def test_precision_threshold_order(score, expected):
    assert PrecisionScheduler.default().select(score).value == expected

def test_int4_quantization_is_symmetric_and_bounded():
    q = fake_quantize(torch.tensor([-20.0, -1.0, 0.0, 1.0, 20.0]), bits=4)
    assert q.values.min().item() >= -7
    assert q.values.max().item() <= 7

def test_observed_high_error_lowers_all_thresholds():
    ctl = EndToEndFeedback.default()
    before = ctl.thresholds.clone()
    ctl.observe(error=0.2, complexity=0.4)
    assert torch.all(ctl.thresholds < before)
```

- [ ] **Step 2: Run precision and feedback tests and verify failures**

Run: `python -m pytest tests/test_precision.py tests/test_feedback.py -q`

- [ ] **Step 3: Implement equations 18-28, per-channel scales, resident packs, and bounded updates**

```python
def precision_score(uncertainty: Tensor, ambiguity: Tensor, confidence: Tensor) -> Tensor:
    return 0.4 * uncertainty + 0.3 * ambiguity + 0.3 * (1.0 - confidence)

def fake_quantize(x: Tensor, bits: int) -> QuantizedTensor:
    limit = 127 if bits == 8 else 7
    scale = x.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / limit
    values = (x.float() / scale).round().clamp(-limit, limit).to(torch.int8)
    return QuantizedTensor(values=values, scale=scale, bits=bits)
```

- [ ] **Step 4: Run precision and feedback tests**

Run: `python -m pytest tests/test_precision.py tests/test_feedback.py -q`

### Task 6: Unified runtime controller

**Files:**
- Create: `src/dsa_repro/controller.py`
- Test: `tests/test_controller.py`

**Interfaces:**
- Consumes: anchor, pruning, precision, and feedback interfaces from Tasks 2-5.
- Produces: `RuntimeController.enter_layer(context) -> LayerPlan`, `RuntimeController.finish_layer(result)`, and `RuntimeController.finish_step(logits)`.

- [ ] **Step 1: Write failing state-machine tests for Algorithm 1 order**

```python
def test_accepted_approximation_does_not_call_full_block(controller, context):
    calls = []
    plan = controller.enter_layer(context.with_high_similarity())
    output = plan.execute(full_block=lambda _: calls.append("full"))
    assert calls == []
    assert output.trace.path == "approximate"

def test_periodic_replay_enlarges_masks_and_clears_fallback_state(controller):
    controller.state.step = 99
    controller.finish_step(torch.zeros(1, 10))
    assert controller.state.force_full_next_step
    assert controller.state.audit_count == 1
```

- [ ] **Step 2: Run controller tests and observe failures**

Run: `python -m pytest tests/test_controller.py -q`

- [ ] **Step 3: Implement the per-layer plan, trace schema, common fallback, and audit scheduling**

```python
@dataclass(frozen=True)
class LayerPlan:
    path: Literal["approximate", "full"]
    mask: MaskDecision
    precision: PrecisionPath
    reason: str

class RuntimeController(nn.Module):
    def enter_layer(self, context: LayerContext) -> LayerPlan:
        approximation = self.anchor.try_layer(context)
        if approximation.accepted:
            return LayerPlan("approximate", MaskDecision.full(context.kv_length), PrecisionPath.FP16, approximation.reason)
        score = self.complexity(context)
        return LayerPlan("full", self.mask_selector.select(score, context.importance, context.kv_length), self.precision.select(context.precision_score), approximation.reason)
```

- [ ] **Step 4: Run all core tests**

Run: `python -m pytest tests/test_signals.py tests/test_anchor.py tests/test_pruning.py tests/test_precision.py tests/test_feedback.py tests/test_controller.py -q`

### Task 7: Hugging Face LLaMA adapter and synthetic smoke runner

**Files:**
- Create: `src/dsa_repro/adapters/__init__.py`
- Create: `src/dsa_repro/adapters/llama.py`
- Create: `src/dsa_repro/generation.py`
- Create: `src/dsa_repro/cli.py`
- Test: `tests/test_llama_adapter.py`
- Test: `tests/test_cli_smoke.py`

**Interfaces:**
- Produces: `DSALlamaAdapter.from_pretrained(config, token)`, `DSALlamaAdapter.from_random_tiny(seed)`, `generate(request) -> GenerationRecord`, and `python -m dsa_repro.cli smoke`.

- [ ] **Step 1: Write failing offline adapter and CLI tests**

```python
def test_tiny_llama_generation_is_deterministic():
    a = DSALlamaAdapter.from_random_tiny(seed=7).generate_ids([1, 2, 3], max_new_tokens=3)
    b = DSALlamaAdapter.from_random_tiny(seed=7).generate_ids([1, 2, 3], max_new_tokens=3)
    assert a.token_ids == b.token_ids
    assert len(a.traces) > 0

def test_smoke_cli_writes_synthetic_manifest(tmp_path):
    result = runner.invoke(app, ["smoke", "--output", str(tmp_path)])
    assert result.exit_code == 0
    assert json.loads((tmp_path / "run_manifest.json").read_text())["result_kind"] == "synthetic_smoke"
```

- [ ] **Step 2: Run adapter tests and observe failures**

Run: `python -m pytest tests/test_llama_adapter.py tests/test_cli_smoke.py -q`

- [ ] **Step 3: Implement restricted-weight loading, frozen parameters, resident tuple KV cache, gathered-key SDPA, generation, and synthetic mode**

```python
@classmethod
def from_pretrained(cls, cfg: ExperimentConfig, token: str | None = None) -> "DSALlamaAdapter":
    if not token:
        raise RuntimeError("HF_TOKEN is required for meta-llama/Llama-2-7b-hf")
    model = LlamaForCausalLM.from_pretrained(cfg.model.model_id, token=token, torch_dtype=torch.float16)
    model.requires_grad_(False).eval()
    return cls(model=model, config=cfg)
```

- [ ] **Step 4: Run adapter tests and smoke command**

Run: `python -m pytest tests/test_llama_adapter.py tests/test_cli_smoke.py -q`

Run: `python -m dsa_repro.cli smoke --output artifacts/smoke`

### Task 8: Calibration traces and auxiliary training

**Files:**
- Create: `src/dsa_repro/calibration.py`
- Create: `src/dsa_repro/training.py`
- Create: `src/dsa_repro/checkpoints.py`
- Test: `tests/test_calibration.py`
- Test: `tests/test_training.py`

**Interfaces:**
- Produces: `collect_calibration(adapter, examples, output_dir)`, `build_pseudo_labels(trace)`, `train_auxiliary(dataset, config, output_dir)`, and versioned checkpoint load/save.

- [ ] **Step 1: Write failing pseudo-label and frozen-base tests**

```python
def test_mapper_label_uses_paper_error_threshold():
    label = build_mapper_label(mapped=torch.ones(4), full=torch.ones(4) * 1.01, epsilon=0.05)
    assert label.acceptable

def test_optimizer_never_receives_base_parameters(tiny_training_bundle):
    optimizer = make_optimizer(tiny_training_bundle.model, tiny_training_bundle.aux, lr=1e-4)
    ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) not in ids for p in tiny_training_bundle.model.parameters())
```

- [ ] **Step 2: Run calibration/training tests and verify failures**

Run: `python -m pytest tests/test_calibration.py tests/test_training.py -q`

- [ ] **Step 3: Implement trace shards, labels for equations 29-32, separate module objectives, AdamW, early stopping, and checkpoint metadata**

```python
def mapper_loss(mapped: Tensor, full: Tensor, eps: float = 1e-6) -> Tensor:
    return ((mapped - full).norm(dim=-1) / full.norm(dim=-1).clamp_min(eps)).mean()

def complexity_target(entropy: Tensor, info_gain: Tensor, dependency: Tensor) -> Tensor:
    return 0.4 * entropy + 0.35 * info_gain + 0.25 * dependency
```

- [ ] **Step 4: Run training tests and a two-batch synthetic train**

Run: `python -m pytest tests/test_calibration.py tests/test_training.py -q`

Run: `python -m dsa_repro.cli train-synthetic --steps 2 --output artifacts/train-smoke`

### Task 9: Dataset manifests, parsers, task metrics, judge, and statistics

**Files:**
- Create: `src/dsa_repro/data.py`
- Create: `src/dsa_repro/metrics.py`
- Create: `src/dsa_repro/judge.py`
- Create: `src/dsa_repro/statistics.py`
- Create: `data/manifests/alpacaeval.json`
- Create: `data/manifests/vicuna80.json`
- Create: `data/manifests/hotpotqa.json`
- Create: `data/manifests/asqa.json`
- Create: `tests/fixtures/benchmark_examples.json`
- Test: `tests/test_data.py`
- Test: `tests/test_metrics.py`
- Test: `tests/test_judge.py`
- Test: `tests/test_statistics.py`

**Interfaces:**
- Produces: `prepare_dataset(name, cache_dir)`, `score_example(task, prediction, reference)`, `JudgeClient.score(...)`, `paired_bootstrap(...)`, and `paired_t_test(...)`.

- [ ] **Step 1: Write failing parser, metric, judge-schema, cache, and paired-statistics tests**

```python
def test_hotpot_normalization_matches_expected():
    score = score_example("hotpotqa", "The Eiffel Tower.", {"answer": "eiffel tower"})
    assert score["exact_match"] == 1.0

def test_judge_requires_environment_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        JudgeClient(model="gpt-4-1106-preview")

def test_paired_bootstrap_is_reproducible():
    a = paired_bootstrap([1, 2, 3], [1, 1, 1], seed=42, samples=100)
    b = paired_bootstrap([1, 2, 3], [1, 1, 1], seed=42, samples=100)
    assert a == b
```

- [ ] **Step 2: Run data/evaluation tests and verify failures**

Run: `python -m pytest tests/test_data.py tests/test_metrics.py tests/test_judge.py tests/test_statistics.py -q`

- [ ] **Step 3: Implement revision-aware manifests, deterministic sampling, metric normalization, Responses API judge requests, dry-run/caching, bootstrap intervals, and paired t-tests**

```python
def paired_t_test(ours: Sequence[float], control: Sequence[float]) -> dict[str, float]:
    if len(ours) != len(control):
        raise ValueError("paired samples must have equal length")
    statistic, pvalue = scipy.stats.ttest_rel(ours, control)
    return {"statistic": float(statistic), "pvalue": float(pvalue), "n": len(ours)}
```

- [ ] **Step 4: Run evaluation tests and judge dry run**

Run: `python -m pytest tests/test_data.py tests/test_metrics.py tests/test_judge.py tests/test_statistics.py -q`

Run: `python -m dsa_repro.cli judge-dry-run --fixture tests/fixtures/benchmark_examples.json --output artifacts/judge-dry-run`

### Task 10: Experiment orchestration, reference results, reports, documentation, and archive

**Files:**
- Create: `src/dsa_repro/experiments.py`
- Create: `src/dsa_repro/reporting.py`
- Create: `scripts/run_full_reproduction.ps1`
- Create: `scripts/run_full_reproduction.sh`
- Create: `paper_reference/table2_stochastic_latency.csv`
- Create: `paper_reference/table7_ablation.csv`
- Create: `paper_reference/table8_gate_sweep.csv`
- Create: `paper_reference/table9_mask_comparison.csv`
- Create: `paper_reference/table10_quality.csv`
- Create: `tests/test_reporting.py`
- Create: `README.md`
- Create: `README_EN.md`
- Create: `docs/PAPER_TO_CODE.md`
- Create: `docs/REPRODUCIBILITY_LIMITS.md`
- Create: `LICENSE`
- Create: `MANIFEST.in`

**Interfaces:**
- Consumes: all prior tasks.
- Produces: `python -m dsa_repro.cli full`, `ablate`, `sweep`, `report`, `verify`, and the final ZIP with SHA-256 checksum.

- [ ] **Step 1: Write failing reference-table and report tests**

```python
def test_gate_sweep_reference_matches_paper():
    frame = pd.read_csv("paper_reference/table8_gate_sweep.csv")
    default = frame.loc[frame["gate_threshold"] == 0.85].iloc[0]
    assert default["approximation_ratio"] == pytest.approx(41.2)
    assert default["latency_ms_per_step"] == pytest.approx(8.32)
    assert default["rouge_l"] == pytest.approx(42.1)

def test_report_refuses_mixed_result_kinds(tmp_path):
    with pytest.raises(ValueError, match="result_kind"):
        build_report([measured_run(tmp_path), synthetic_run(tmp_path)], tmp_path / "report")
```

- [ ] **Step 2: Run reporting tests and verify failures**

Run: `python -m pytest tests/test_reporting.py -q`

- [ ] **Step 3: Implement resumable stage orchestration, CUDA timing, ablation matrix, gate sweep, measured-data plotting, paper-reference CSVs, and report validation**

```python
STAGES = ("preflight", "prepare", "calibrate", "train", "evaluate", "latency", "ablate", "sweep", "report")

def validate_result_kinds(manifests: Sequence[dict[str, object]]) -> str:
    kinds = {str(item["result_kind"]) for item in manifests}
    if len(kinds) != 1:
        raise ValueError(f"result_kind mismatch: {sorted(kinds)}")
    return kinds.pop()
```

- [ ] **Step 4: Write detailed documentation**

README sections must cover scientific scope, paper-to-code mapping, archive tree, exact A100 environment creation, credentials, model license/access, dataset preparation, calibration, training, each benchmark, judge compatibility, latency methodology, every ablation, gate sweep, statistics, output schemas, resume behavior, troubleshooting, limitations, expected runtime/storage, and citation.

- [ ] **Step 5: Run the complete verification suite**

Run: `python -m pytest -q`

Run: `python -m ruff check src tests`

Run: `python -m compileall -q src scripts`

Run: `python -m dsa_repro.cli smoke --output artifacts/final-smoke`

Run: `python -m dsa_repro.cli verify --run-dir artifacts/final-smoke`

- [ ] **Step 6: Build and inspect the archive**

Run: `python -m build`

Run: `python scripts/package_release.py --source . --output ../Dynamic_Semantic_Adaptation_Reproduction_20260729.zip`

Run: `python scripts/verify_archive.py ../Dynamic_Semantic_Adaptation_Reproduction_20260729.zip`

Expected: archive verification reports no secrets, excluded caches, missing required files, checksum mismatches, or failing extracted smoke tests.
