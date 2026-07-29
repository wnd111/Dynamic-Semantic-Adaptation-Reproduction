from __future__ import annotations

import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .adapters.llama import DSALlamaAdapter
from .calibration import CalibrationTrace, save_trace_shard
from .config import ExperimentConfig
from .metrics import score_example
from .pipeline import extract_prompt_reference, read_jsonl, summarize_metrics
from .precision import fake_quantize
from .provenance import atomic_write_json


def load_runtime(
    config: ExperimentConfig,
) -> tuple[DSALlamaAdapter, Any]:
    """Load the gated base model and matching tokenizer using the runner's HF_TOKEN."""
    from transformers import AutoTokenizer

    token = os.getenv("HF_TOKEN")
    adapter = DSALlamaAdapter.from_pretrained(config, token=token)
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, token=token)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return adapter, tokenizer


def _cpu_half(value: Tensor) -> Tensor:
    return value.detach().to(device="cpu", dtype=torch.float16)


def collect_calibration_traces(
    config: ExperimentConfig,
    output_dir: Path,
    *,
    max_sequences: int | None = None,
    shard_sequences: int = 8,
    auxiliary_data: Path | None = None,
) -> list[Path]:
    """Run the frozen full model on C4 and persist last-token layer supervision."""
    from datasets import load_dataset
    from transformers import AutoTokenizer, LlamaForCausalLM

    if shard_sequences < 1:
        raise ValueError("shard_sequences must be positive")
    count = max_sequences or config.training.calibration_sequences
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for calibration")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_id, token=token)
    model = LlamaForCausalLM.from_pretrained(
        config.model.model_id,
        token=token,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).to("cuda")
    model.requires_grad_(False).eval()
    dataset = load_dataset(
        config.training.calibration_dataset,
        "en",
        split=config.training.calibration_split,
        streaming=True,
    )
    traces: list[CalibrationTrace] = []
    shard_paths: list[Path] = []
    sequences_in_shard = 0
    auxiliary_rows = read_jsonl(auxiliary_data) if auxiliary_data is not None else []

    def texts() -> Iterable[str]:
        c4_consumed = 0
        for row in dataset:
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            yield text
            c4_consumed += 1
            if c4_consumed == count:
                break
        if c4_consumed != count:
            raise RuntimeError(f"C4 stream ended after {c4_consumed} usable rows; expected {count}")
        for row in auxiliary_rows:
            conversations = row.get("conversations", [])
            if not isinstance(conversations, list):
                raise ValueError("ShareGPT conversations must be a list")
            turns = []
            for turn in conversations:
                if not isinstance(turn, dict):
                    continue
                speaker = str(turn.get("from", "unknown"))
                value = str(turn.get("value", ""))
                turns.append(f"{speaker}: {value}")
            yield "\n".join(turns)

    expected = count + len(auxiliary_rows)
    consumed = 0
    for text in texts():
        encoded = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=config.training.calibration_context_length,
            padding="max_length",
        ).to("cuda")
        with torch.inference_mode():
            outputs = model(
                **encoded,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
        hidden_states = outputs.hidden_states
        attentions = outputs.attentions
        if hidden_states is None or attentions is None:
            raise RuntimeError("model did not return calibration hidden states/attentions")
        logits_prefix = _cpu_half(outputs.logits[:, -2, :])
        logits_next = _cpu_half(outputs.logits[:, -1, :])
        for layer_idx in range(1, len(attentions)):
            layer_input = _cpu_half(hidden_states[layer_idx][:, -1:, :])
            anchor = _cpu_half(hidden_states[layer_idx - 1][:, -1:, :])
            full = _cpu_half(hidden_states[layer_idx + 1][:, -1:, :])
            low = fake_quantize(full, bits=4).dequantize(torch.float16)
            traces.append(
                CalibrationTrace(
                    layer_idx=layer_idx,
                    layer_input=layer_input,
                    anchor_hidden=anchor,
                    mapped_hidden=anchor,
                    corrected_hidden=anchor,
                    full_hidden=full,
                    logits_prefix=logits_prefix,
                    logits_next=logits_next,
                    attention=_cpu_half(attentions[layer_idx][:, :, -1:, :]),
                    low_precision_output=low,
                    fp32_output=full.float(),
                )
            )
        consumed += 1
        sequences_in_shard += 1
        if sequences_in_shard == shard_sequences or consumed == expected:
            path = save_trace_shard(traces, output_dir / f"trace-{len(shard_paths):05d}.pt")
            shard_paths.append(path)
            traces = []
            sequences_in_shard = 0
        if consumed == expected:
            break
    if consumed != expected:
        raise RuntimeError(f"trace collection ended after {consumed} rows; expected {expected}")
    atomic_write_json(
        output_dir / "calibration_record.json",
        {
            "schema_version": 1,
            "dataset": config.training.calibration_dataset,
            "split": config.training.calibration_split,
            "c4_sequences": count,
            "auxiliary_sequences": len(auxiliary_rows),
            "sequences": consumed,
            "auxiliary_data": str(auxiliary_data.resolve()) if auxiliary_data else None,
            "context_length": config.training.calibration_context_length,
            "trace_shards": [path.name for path in shard_paths],
            "dataset_fingerprint_note": (
                "record the datasets cache fingerprint from the run manifest"
            ),
        },
    )
    return shard_paths


def evaluate_prepared(
    *,
    task: str,
    prepared_path: Path,
    adapter: DSALlamaAdapter,
    tokenizer: Any,
    output_path: Path,
    context_length: int,
    max_new_tokens: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows = read_jsonl(prepared_path)
    if max_samples is not None:
        rows = rows[:max_samples]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics: list[dict[str, float]] = []
    traces = 0
    approximate_traces = 0
    precision_counts: dict[str, int] = {}
    with output_path.open("w", encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            prompt, reference = extract_prompt_reference(task, row)
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)[-context_length:]
            record = adapter.generate_ids(prompt_ids, max_new_tokens=max_new_tokens)
            prediction = tokenizer.decode(record.token_ids, skip_special_tokens=True).strip()
            example_metrics: dict[str, float] = {}
            if task != "alpacaeval" and reference.get("answer", reference):
                try:
                    example_metrics = score_example(task, prediction, reference)
                    metrics.append(example_metrics)
                except ValueError:
                    example_metrics = {}
            traces += len(record.traces)
            for trace in record.traces:
                approximate_traces += int(trace.get("path") == "approximate")
                precision = str(trace.get("precision"))
                precision_counts[precision] = precision_counts.get(precision, 0) + 1
            payload = {
                "index": index,
                "task": task,
                "prompt": prompt,
                "prediction": prediction,
                "reference": reference,
                "metrics": example_metrics,
                "generation": record.to_dict(),
            }
            stream.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "task": task,
        "samples": len(rows),
        "metrics": summarize_metrics(metrics),
        "layer_traces": traces,
        "approximation_ratio": 100.0 * approximate_traces / traces if traces else 0.0,
        "precision_counts": precision_counts,
        "result_kind": adapter.result_kind,
    }
    atomic_write_json(output_path.with_suffix(".summary.json"), summary)
    return summary


def benchmark_adapter(
    adapter: DSALlamaAdapter,
    prompt_ids: list[int],
    *,
    warmup_steps: int,
    measurement_steps: int,
) -> dict[str, float]:
    """Measure one-token decode latency; synchronize CUDA around every sample."""
    if warmup_steps < 0 or measurement_steps < 1:
        raise ValueError("invalid benchmark step counts")

    def synchronize() -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    for _ in range(warmup_steps):
        adapter.generate_ids(prompt_ids, max_new_tokens=1)
    timings: list[float] = []
    for _ in range(measurement_steps):
        synchronize()
        start = time.perf_counter()
        adapter.generate_ids(prompt_ids, max_new_tokens=1)
        synchronize()
        timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    return {
        "mean_ms": sum(timings) / len(timings),
        "median_ms": ordered[len(ordered) // 2],
        "min_ms": ordered[0],
        "max_ms": ordered[-1],
        "samples": float(len(timings)),
    }


def find_trace_shards(directory: Path) -> list[Path]:
    paths = sorted(Path(directory).glob("trace-*.pt"))
    if not paths:
        raise FileNotFoundError(f"no trace-*.pt shards found in {directory}")
    return paths


def mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise ValueError("cannot average an empty iterable")
    return sum(materialized) / len(materialized)
