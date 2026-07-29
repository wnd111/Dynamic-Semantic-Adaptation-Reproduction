# **Dynamic Semantic Adaptation for Efficient Large Language Model Inference**

The target hardware is a **single NVIDIA A100 80GB GPU**. This codebase covers dynamic anchor approximation, predictive attention pruning, adaptive precision scheduling, local refresh and end-to-end feedback, calibration-data generation, auxiliary-module training, evaluation on four benchmarks, ablation studies, threshold sweeps, latency measurement, report generation, checkpointing, and release-package validation.

## 1. Quick Start

On a Linux machine with an A100 GPU:

```bash
conda env create -f environment-a100.yml
conda activate dsa-repro-a100
pip install -e .

export HF_TOKEN="provided-by-runner"
export OPENAI_API_KEY="provided-by-runner"

dsa-repro preflight --config configs/paper_a100.yaml
dsa-repro smoke --output artifacts/smoke
dsa-repro train-synthetic --output artifacts/train-smoke
pytest -q
```

On PowerShell:

```powershell
conda env create -f environment-a100.yml
conda activate dsa-repro-a100
pip install -e .

$env:HF_TOKEN = "provided-by-runner"
$env:OPENAI_API_KEY = "provided-by-runner"

dsa-repro preflight --config configs/paper_a100.yaml
dsa-repro smoke --output artifacts/smoke
pytest -q
```

The package contains no real credentials and never writes credentials to a manifest. A manifest records only whether each credential is `present` or `absent`. Before accessing the LLaMA 2 weights, the runner must accept the license on Hugging Face and provide an authorized `HF_TOKEN`. The OpenAI judge reads credentials only from `OPENAI_API_KEY`.

## 2. Repository Structure

```text
configs/
data/manifests/
docs/
paper_reference/
scripts/
src/dsa_repro/
  adapters/llama.py
  anchor.py
  pruning.py
  precision.py
  feedback.py
  controller.py
  calibration.py
  training.py
  workflow.py
  judge.py
  reporting.py
tests/
```

## 3. Paper Protocol

`configs/paper_a100.yaml` fixes the experimental settings that can be verified from the paper PDF:

| Item | Paper protocol |
|---|---:|
| Base model | `meta-llama/Llama-2-7b-hf` |
| GPU | One NVIDIA A100 80GB GPU |
| PyTorch / CUDA / cuDNN | 2.1.0 / 12.1 / 8.9.2 |
| Transformers | 4.36.0 |
| Batch size / context length | 1 / 512 |
| Latency protocol | 10 warm-up runs and 1,000 synchronized measurements |
| Anchor gate / drift / maximum chain length | 0.85 / 0.15 / 3 |
| Pruning complexity thresholds | High: 0.7; low: 0.3 |
| Window / top-k | 0.9 / 0.5 |
| Precision thresholds | 0.7 / 0.5 / 0.3 |
| Local audit / end-to-end replay | Every 20 / 100 steps |
| C4 calibration | 2,048 samples, length 512, seed 42 |
| Auxiliary training | AdamW, learning rate 1e-4, weight decay 0.01, batch size 64, up to 3 epochs |
| Evaluation samples | AlpacaEval: 805; Vicuna-80: 80; HotpotQA: 405; ASQA: 948 |

The environment file pins the major dependency versions. cuDNN is provided by the NVIDIA/PyTorch CUDA runtime, and the preflight report records the version detected at runtime. The LLaMA implementation in Transformers 4.36.0 can be checked against the [official source code](https://github.com/huggingface/transformers/blob/v4.36.0/src/transformers/models/llama/modeling_llama.py).

## 4. One-Command Full Workflow

Save a copy of the original configuration before editing any parameters. Run the full workflow with:

```bash
bash scripts/run_full_a100.sh
```

Or on PowerShell:

```powershell
./scripts/run_full_a100.ps1
```

The script runs the following stages in order: preflight checks, preparation of four evaluation datasets, C4 calibration, auxiliary training, generation and metrics for all four datasets, latency measurement, ablation studies, gate sweeps, and report generation. Outputs are written to `artifacts/paper-a100/` by default.

You can also generate a step-by-step command manifest first:

```bash
dsa-repro plan-full \
  --config configs/paper_a100.yaml \
  --run-dir artifacts/paper-a100 \
  --output artifacts/runbook.json
```

### 4.1 Data Preparation

```bash
for name in alpacaeval vicuna80 hotpotqa asqa sharegpt512; do
  dsa-repro prepare-data "$name" --output artifacts/paper-a100/data
done
```

Each manifest records the source, commit or version notes, split, sample count, sampling seed, required fields, and license. The preparation stage generates `prepared.jsonl`, a SHA-256 checksum, and a data record. The data interfaces are based on the official [AlpacaEval](https://huggingface.co/datasets/tatsu-lab/alpaca_eval), [HotpotQA](https://huggingface.co/datasets/hotpotqa/hotpot_qa), and [ASQA](https://huggingface.co/datasets/din0s/asqa) dataset pages.

The public Vicuna-80 file contains only questions and has no unified reference answers. This package therefore generates answers and traces. To compute metrics, add an `answer` field to the prepared JSONL file.

### 4.2 Calibration and Auxiliary Training

```bash
dsa-repro calibrate \
  --config configs/paper_a100.yaml \
  --auxiliary-data artifacts/paper-a100/data/sharegpt512/prepared.jsonl \
  --output artifacts/paper-a100/calibration

dsa-repro train \
  --config configs/paper_a100.yaml \
  --traces artifacts/paper-a100/calibration \
  --output artifacts/paper-a100/checkpoints
```

Calibration runs a frozen, complete LLaMA 2 model on the C4 validation split and records layer inputs, anchors from the preceding full layer, full-layer outputs, last-token attention, logits, and low-precision errors. To limit disk use, data is divided into shards of eight sequences, and only last-token layer supervision is stored.

For a small-scale pipeline check, add:

```bash
dsa-repro calibrate --max-sequences 2 --shard-sequences 1 --output artifacts/calibration-debug
dsa-repro train --traces artifacts/calibration-debug --max-epochs 1 --output artifacts/train-debug
```

These outputs are still marked as `measured`, but their sample counts are recorded in the manifest and they must not be used for comparisons with the paper's reported values.

### 4.3 Quality Evaluation

```bash
dsa-repro evaluate \
  --config configs/paper_a100.yaml \
  --checkpoint artifacts/paper-a100/checkpoints/auxiliary.pt \
  --data-root artifacts/paper-a100/data \
  --output artifacts/paper-a100/evaluation
```

HotpotQA reports exact match, token F1, and ROUGE-L. ASQA reports string match, disambiguation F1, and ROUGE-L. Vicuna-80 reports exact match and F1 when reference answers are available. Each JSONL record stores the prompt, prediction, reference, metrics, runtime backend, Transformers version, and the per-layer path, mask, and precision trace.

Run the AlpacaEval LLM-as-a-judge stage separately to avoid accidental paid API calls:

```bash
# Generate requests without calling the API first.
dsa-repro judge-results \
  --predictions artifacts/paper-a100/evaluation/alpacaeval.jsonl \
  --config configs/current_compatible.yaml \
  --output artifacts/judge-dry --dry-run

# Remove --dry-run only after reviewing the requests.
```

### 4.4 Latency, Ablation, and Sensitivity

```bash
dsa-repro latency --config configs/paper_a100.yaml \
  --checkpoint artifacts/paper-a100/checkpoints/auxiliary.pt \
  --output artifacts/paper-a100/latency

dsa-repro ablate --config configs/paper_a100.yaml \
  --checkpoint artifacts/paper-a100/checkpoints/auxiliary.pt \
  --data-root artifacts/paper-a100/data \
  --output artifacts/paper-a100/ablation

dsa-repro sweep --config configs/paper_a100.yaml \
  --checkpoint artifacts/paper-a100/checkpoints/auxiliary.pt \
  --data-root artifacts/paper-a100/data \
  --output artifacts/paper-a100/gate-sweep
```

Each ablation switch disables only its corresponding mechanism:

| Variant | Anchor approximation | Pruning | Precision scheduling |
|---|---:|---:|---:|
| `full` | On | On | On |
| `no-approximation` | Off | On | On |
| `no-pruning` | On | Off | On |
| `fp16-only` | On | On | Fixed to FP16 |

The latency command calls CUDA synchronization before and after each sample, performs 10 warm-up runs, then collects 1,000 measurements and reports the mean, median, minimum, and maximum.

## 5. Outputs and Traceability

Every formal stage includes:

- `run_manifest.json`: a configuration snapshot, package version, CUDA/GPU details, Git commit, dirty state, and credential-presence indicators;
- primary result files in JSON, JSONL, or CSV format, including `result_kind`;
- calibration and checkpoint metadata, including the training epoch, loss, and absolute input-shard paths;
- a judge cache containing hashed requests and responses to avoid duplicate charges on reruns;
- `stage_state.json`, which allows `ExperimentRunner` to determine from SHA-256 file hashes whether a stage is still valid.
