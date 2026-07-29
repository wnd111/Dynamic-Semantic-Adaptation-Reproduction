# Dynamic Semantic Adaptation Full Reproduction Design

## Objective

Build a self-contained reproduction package for *Dynamic Semantic Adaptation for Efficient Large Language Model Inference*. The target machine is one NVIDIA A100 80GB GPU. The package must implement the paper's three runtime decisions, train the frozen-base auxiliary modules, evaluate all four reported benchmarks, reproduce the ablations and gate sweep, regenerate the reported tables and Figure 2, and provide a detailed Chinese README.

The base model remains frozen. `HF_TOKEN` and `OPENAI_API_KEY` are read from the environment at runtime; no credential or model weight is included in the archive.

## Considered Approaches

### 1. Minimal algorithm demonstration

Implement the equations on synthetic tensors and provide one Hugging Face generation example. This is easy to audit and test but cannot reproduce the paper's calibration, four-dataset evaluation, paired statistics, or latency tables. It is insufficient for the requested scope.

### 2. Paper-faithful reference system with an optimized A100 path - selected

Separate the implementation into a deterministic, testable PyTorch reference controller and an A100 execution adapter for LLaMA-2-7B. The same configuration and controller decisions drive calibration, auxiliary training, generation, evaluation, ablations, and plotting. Fast execution uses single-token decoding, resident KV caches, reduced-key attention, mixed precision, and optional packed weight backends. A CPU synthetic mode tests the entire orchestration without downloading restricted weights.

This approach preserves auditability while still providing a practical full reproduction workflow. It also makes limitations caused by unavailable original checkpoints or unpublished custom kernels explicit.

### 3. New custom CUDA inference engine

Reimplement the model, KV allocator, sparse attention, INT8/INT4 GEMM, and fused dispatch in CUDA/Triton. This could best match absolute latency, but the paper does not specify enough kernel details to reproduce the exact implementation. It would greatly increase maintenance and validation risk without improving scientific traceability. It is outside the justified scope of a paper-only reproduction.

## Reproducibility Boundary

The archive is a clean-room reconstruction from the supplied 17-page manuscript and its equations, tables, algorithm, appendix, and configuration. It does not claim to contain the authors' unreleased source, trained auxiliary weights, calibration examples, judge responses, or exact CUDA kernels.

The package distinguishes three result classes:

- `paper_reference`: numbers transcribed from Tables 2 and 7-10 and Figure 2.
- `measured`: outputs produced by the included scripts on the user's machine.
- `synthetic_smoke`: deterministic checks that validate orchestration only and are never presented as paper reproduction results.

## Architecture

### Configuration and provenance

Typed configuration objects load versioned YAML files. The default configuration contains the fixed values from Tables 3, 5, and 6: LLaMA-2-7B, 512-token calibration context, 2,048 calibration sequences, gate threshold 0.85, drift threshold 0.15, maximum reuse count 3, mask thresholds 0.7 and 0.3, end-to-end audit interval 100, local audit interval 20, precision thresholds 0.7/0.5/0.3, and five paired seeds. Every run writes the resolved configuration, seed, package versions, GPU information, Git state when available, and timestamps.

### Core controller

Pure PyTorch modules implement the manuscript's execution order:

1. Pool the current and candidate-anchor hidden states, project them, and compute normalized cosine similarity.
2. If the gate accepts the candidate, map the anchor through a gated residual mapper.
3. Run the drift detector. Apply one residual correction and recheck; otherwise fall back to the full layer.
4. Synthesize the current layer's K/V entries from the accepted approximated hidden state so all later layers retain a resident cache.
5. If full execution is required, predict complexity from semantic entropy, information gain, and dependency span; select full, 90% window, or deterministic top-50% attention paths.
6. Combine confidence, uncertainty, and ambiguity into the precision score and choose FP32 accumulation, FP16/BF16, INT8, or INT4 according to the ordered thresholds.
7. Apply local paired FP32 precision audits and periodic end-to-end replay. Update only controller thresholds; never update base-model weights online.
8. Enforce the common fallback after three consecutive accepted approximations or after an end-to-end divergence event.

Each module exposes tensor-level diagnostics so unit tests and analysis scripts can verify the exact decision path.

### LLaMA-2 adapter

The adapter loads `meta-llama/Llama-2-7b-hf` with `HF_TOKEN`, freezes it, and executes deterministic batch-size-one autoregressive decoding. It owns the resident KV cache and invokes the core controller at each decoder layer. The adapter provides two execution backends:

- `reference`: portable PyTorch implementation for correctness and analysis.
- `a100`: CUDA mixed-precision path using PyTorch scaled-dot-product attention on gathered resident keys and optional quantized linear backends. Unsupported packed-kernel combinations fail clearly or fall back according to configuration; they are never silently reported as quantized latency results.

### Calibration and auxiliary training

The calibration command samples 2,048 C4 validation sequences with seed 42 and separately prepares 512 ShareGPT conversations. Frozen full-model traces produce hidden states, logits, attention probabilities, full-block targets, and FP32 reference outputs.

Training commands implement Appendix B:

- residual mapper with normalized residual loss;
- projection and gate with mapper labels, cross-entropy, and acceptable error threshold 0.05;
- residual corrector using one-correction failures;
- drift detector using the 0.05 post-correction target;
- complexity predictor with MSE targets from entropy, Jensen-Shannon information gain, and matched dependency span;
- confidence head with BCE against the low-precision acceptability label;
- uncertainty head with MSE against normalized semantic entropy.

AdamW uses learning rate 1e-4, weight decay 0.01, batch size 64, at most three epochs, validation-only model selection, early stopping, and no base-model gradients.

### Data and evaluation

Dataset preparation scripts download or validate AlpacaEval prompts, the 80-question Vicuna benchmark, HotpotQA distractor-development samples, and the ASQA development split. Immutable manifests record upstream identifiers, revisions when available, splits, sample selection, and hashes. Raw data and model weights are excluded from the ZIP.

Evaluation commands reproduce the paper's task-specific metrics:

- AlpacaEval: single-response GPT-4-Turbo score from 1 to 10;
- Vicuna-80: exact match and token F1;
- HotpotQA: exact match, token F1, and ROUGE-L;
- ASQA: short-answer exact match, disambiguation F1, and ROUGE-L.

The judge client reads only `OPENAI_API_KEY`, uses temperature 0, validates the response schema, caches request hashes, retries transient failures, and records the exact prompt and model identifier. A dry-run mode prepares requests without sending them. The paper-exact profile retains `gpt-4-1106-preview`; a separate current-compatible profile uses the official replacement model because the paper's snapshot is deprecated. Results produced by different judge models are labeled non-comparable rather than mixed.

### Experiments and reports

One top-level command orchestrates environment checks, data preparation, calibration, auxiliary training, quality evaluation, latency measurement, one-module-at-a-time ablations, gate threshold sweep, learned-versus-training-free mask comparisons, paired bootstrap intervals, paired t-tests, and plot/table generation.

All experiment outputs are written to timestamped directories. Plotting code reads measured CSV/JSON files, never constants embedded in plotting logic. Separate commands render the transcribed paper-reference tables for comparison.

## Error Handling

- Missing `HF_TOKEN`, `OPENAI_API_KEY`, restricted model access, CUDA, A100 capability, dataset fields, checkpoints, or optional kernels produce actionable errors.
- Configuration validation rejects inconsistent thresholds, unsupported dtypes, nonresident KV mode, batch sizes other than one for the main protocol, and accidental use of test tables for calibration.
- Generated outputs are written atomically. Existing run directories are not overwritten.
- Resume state includes completed stage checksums so interrupted full runs can continue safely.
- Any fallback from the requested optimized path is recorded in the run manifest and excluded from latency comparisons unless explicitly allowed.

## Testing and Verification

The test suite follows red-green development and contains:

- equation tests for normalized cosine similarity, entropy, Jensen-Shannon information gain, ambiguity, local precision error, PPR divergence, and threshold updates;
- state-machine tests for anchor release timing, maximum approximation chains, correction/fallback, audit timing, and deterministic mask selection;
- quantization tests for scales, clipping, packing metadata, and dequantization error bounds;
- adapter tests using a tiny randomly initialized LLaMA configuration without external downloads;
- dataset parser and metric tests with small fixtures;
- judge request/parser tests with recorded local responses and no network calls;
- CLI smoke tests covering prepare, calibrate, train, evaluate, ablate, sweep, report, and the synthetic end-to-end path;
- schema and provenance tests ensuring synthetic, reference, and measured results cannot be mixed.

Before packaging, the full local test suite, static import/compile checks, CLI help checks, synthetic end-to-end run, archive listing, checksum generation, and extraction/retest check must pass. GPU-only commands that cannot be executed in the current environment are marked as requiring the target A100; the README gives exact commands and expected artifacts without claiming unperformed measurements.

## Deliverables

The final directory and ZIP contain source code, tests, YAML configurations, dataset manifests, environment files, command-line scripts, reference-result files, reporting utilities, a Chinese detailed README, an English quick-start, a paper-to-code map, a limitations statement, and checksums. They exclude credentials, restricted weights, downloaded datasets, generated caches, and local virtual environments.
