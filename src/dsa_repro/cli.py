from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import typer

from .adapters.llama import DSALlamaAdapter
from .calibration import load_trace_shard
from .checkpoints import load_checkpoint, save_checkpoint
from .config import load_config
from .data import prepare_dataset
from .experiments import build_runbook
from .experiments import preflight as run_preflight
from .judge import JudgeClient
from .pipeline import configure_variant, read_jsonl
from .provenance import atomic_write_json, write_manifest
from .reporting import build_report
from .training import AuxiliaryModules, train_synthetic_steps, train_trace_shards
from .workflow import (
    benchmark_adapter,
    collect_calibration_traces,
    evaluate_prepared,
    find_trace_shards,
    load_runtime,
)

app = typer.Typer(
    name="dsa-repro",
    help="Reproduce Dynamic Semantic Adaptation experiments.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Dynamic Semantic Adaptation reproduction commands."""


def _empty_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise typer.BadParameter(f"output directory is not empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


@app.command()
def smoke(
    output: Path = typer.Option(Path("artifacts/smoke"), help="Fresh output directory."),
    max_new_tokens: int = typer.Option(3, min=1, help="Number of greedy tokens."),
    seed: int = typer.Option(7, help="Tiny-model initialization seed."),
) -> None:
    """Run an offline tiny-LLaMA end-to-end control-flow check."""
    _empty_output(output)
    config = load_config(Path("configs/paper_a100.yaml"))
    write_manifest(output, config, "synthetic_smoke")
    record = DSALlamaAdapter.from_random_tiny(seed=seed).generate_ids(
        [1, 5, 9],
        max_new_tokens=max_new_tokens,
    )
    atomic_write_json(output / "generation.json", record.to_dict())
    typer.echo(f"synthetic smoke run written to {output}")


@app.command("train-synthetic")
def train_synthetic(
    output: Path = typer.Option(Path("artifacts/train-smoke"), help="Fresh output directory."),
    steps: int = typer.Option(2, min=1, help="Number of optimizer steps."),
    seed: int = typer.Option(5, help="Synthetic training seed."),
) -> None:
    """Exercise every auxiliary objective without model or dataset downloads."""
    _empty_output(output)
    config = load_config(Path("configs/paper_a100.yaml"))
    write_manifest(output, config, "synthetic_smoke")
    base = __import__("torch").nn.Linear(16, 16)
    auxiliary = AuxiliaryModules(hidden_size=16)
    losses = train_synthetic_steps(base, auxiliary, steps=steps, seed=seed)
    save_checkpoint(
        output / "auxiliary.pt",
        auxiliary,
        metadata={"result_kind": "synthetic_smoke", "steps": steps, "seed": seed},
    )
    atomic_write_json(output / "training.json", {"losses": losses, "steps": steps})
    typer.echo(f"synthetic training run written to {output}")


@app.command("prepare-data")
def prepare_data(
    name: str = typer.Argument(
        ...,
        help="alpacaeval, vicuna80, hotpotqa, asqa, or sharegpt512",
    ),
    output: Path = typer.Option(Path("data/prepared"), help="Fresh dataset output directory."),
) -> None:
    """Download, validate, deterministically select, and hash one benchmark."""
    allowed = {"alpacaeval", "vicuna80", "hotpotqa", "asqa", "sharegpt512"}
    if name not in allowed:
        raise typer.BadParameter(f"name must be one of: {', '.join(sorted(allowed))}")
    prepared = prepare_dataset(Path("data/manifests") / f"{name}.json", output / name)
    typer.echo(f"prepared dataset written to {prepared}")


@app.command("judge-dry-run")
def judge_dry_run(
    fixture: Path = typer.Option(..., exists=True, dir_okay=False, help="JSON list of prompts."),
    output: Path = typer.Option(Path("artifacts/judge-dry-run"), help="Fresh output directory."),
    model: str = typer.Option("gpt-4-1106-preview", help="Recorded judge model identifier."),
) -> None:
    """Materialize judge requests without calling the OpenAI API."""
    _empty_output(output)
    rows = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise typer.BadParameter("fixture must contain a JSON list")
    judge = JudgeClient(
        model=model,
        cache_dir=output,
        allow_missing_key_for_dry_run=True,
    )
    requests = []
    for row in rows:
        if row.get("task") != "alpacaeval":
            continue
        requests.append(
            judge.score(
                str(row["instruction"]),
                str(row["prediction"]),
                dry_run=True,
            )
        )
    atomic_write_json(output / "index.json", {"requests": requests, "count": len(requests)})
    typer.echo(f"{len(requests)} judge request(s) written to {output}")


@app.command()
def preflight(
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/preflight.json")),
    strict: bool = typer.Option(True, help="Exit non-zero when A100/HF_TOKEN requirements fail."),
) -> None:
    """Validate the machine, versions, GPU, and credential presence."""
    cfg = load_config(config)
    report = run_preflight(cfg, strict=False)
    atomic_write_json(output, report.to_dict())
    for warning in report.warnings:
        typer.echo(f"warning: {warning}")
    for issue in report.issues:
        typer.echo(f"issue: {issue}")
    if strict and not report.ok:
        raise typer.Exit(code=2)
    typer.echo(f"preflight report written to {output}")


@app.command("plan-full")
def plan_full(
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    run_dir: Path = typer.Option(Path("artifacts/paper-a100")),
    output: Path = typer.Option(Path("artifacts/runbook.json")),
) -> None:
    """Write the complete, explicit command runbook without launching costly jobs."""
    atomic_write_json(output, build_runbook(config, run_dir))
    typer.echo(f"runbook written to {output}")


@app.command()
def calibrate(
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/calibration")),
    max_sequences: int | None = typer.Option(None, min=1, help="Debug override; omit for 2048."),
    shard_sequences: int = typer.Option(8, min=1),
    auxiliary_data: Path | None = typer.Option(
        None,
        exists=True,
        dir_okay=False,
        help="Prepared deterministic ShareGPT-512 proxy JSONL.",
    ),
) -> None:
    """Collect frozen-full-model C4 supervision traces."""
    _empty_output(output)
    cfg = load_config(config)
    write_manifest(output, cfg, "measured")
    shards = collect_calibration_traces(
        cfg,
        output,
        max_sequences=max_sequences,
        shard_sequences=shard_sequences,
        auxiliary_data=auxiliary_data,
    )
    typer.echo(f"wrote {len(shards)} calibration shard(s) to {output}")


@app.command("train")
def train_auxiliary(
    traces: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/checkpoints")),
    max_epochs: int | None = typer.Option(None, min=1),
) -> None:
    """Train only DSA auxiliary modules; the LLaMA base remains frozen."""
    from transformers import AutoConfig

    _empty_output(output)
    cfg = load_config(config)
    model_cfg = AutoConfig.from_pretrained(cfg.model.model_id, token=os.getenv("HF_TOKEN"))
    torch = __import__("torch")
    auxiliary = AuxiliaryModules(
        hidden_size=int(model_cfg.hidden_size),
        residual_blend=cfg.anchor.residual_blend,
    )
    if torch.cuda.is_available():
        auxiliary = auxiliary.to("cuda")
    shard_paths = find_trace_shards(traces)
    # Validate the first shard early, before a multi-hour optimization job.
    load_trace_shard(shard_paths[0])
    epochs = max_epochs or cfg.training.max_epochs
    losses = train_trace_shards(
        shard_paths,
        auxiliary,
        epochs=epochs,
        batch_size=cfg.training.batch_size,
        learning_rate=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        seed=cfg.training.seed,
    )
    checkpoint = save_checkpoint(
        output / "auxiliary.pt",
        auxiliary,
        metadata={
            "model_id": cfg.model.model_id,
            "epochs": epochs,
            "losses": losses,
            "trace_shards": [str(path.resolve()) for path in shard_paths],
            "result_kind": "measured",
        },
    )
    write_manifest(output, cfg, "measured")
    atomic_write_json(output / "training.json", {"epoch_losses": losses})
    typer.echo(f"checkpoint written to {checkpoint}")


def _install_checkpoint(adapter: DSALlamaAdapter, checkpoint: Path, residual_blend: float) -> None:
    auxiliary = AuxiliaryModules(
        hidden_size=int(adapter.model.config.hidden_size),
        residual_blend=residual_blend,
    )
    load_checkpoint(checkpoint, auxiliary)
    adapter.install_auxiliary(auxiliary)


def _evaluate_loaded(
    *,
    cfg: object,
    adapter: DSALlamaAdapter,
    tokenizer: object,
    data_root: Path,
    output: Path,
    task: str,
    max_samples: int | None,
    max_new_tokens: int | None,
) -> list[dict[str, object]]:
    tasks = list(cfg.evaluation.datasets) if task == "all" else [task]
    summaries = []
    for name in tasks:
        prepared = data_root / name / "prepared.jsonl"
        if not prepared.is_file():
            raise typer.BadParameter(f"prepared dataset is missing: {prepared}")
        adapter.reset_runtime()
        summaries.append(
            evaluate_prepared(
                task=name,
                prepared_path=prepared,
                adapter=adapter,
                tokenizer=tokenizer,
                output_path=output / f"{name}.jsonl",
                context_length=cfg.runtime.context_length,
                max_new_tokens=max_new_tokens or cfg.runtime.max_new_tokens,
                max_samples=max_samples,
            )
        )
    return summaries


@app.command()
def evaluate(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    data_root: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/evaluation")),
    task: str = typer.Option("all"),
    variant: str = typer.Option("full"),
    max_samples: int | None = typer.Option(None, min=1),
    max_new_tokens: int | None = typer.Option(None, min=1),
) -> None:
    """Generate benchmark answers and compute deterministic task metrics."""
    _empty_output(output)
    cfg = load_config(config)
    adapter, tokenizer = load_runtime(cfg)
    _install_checkpoint(adapter, checkpoint, cfg.anchor.residual_blend)
    configure_variant(adapter.controller, variant)
    write_manifest(output, cfg, "measured")
    summaries = _evaluate_loaded(
        cfg=cfg,
        adapter=adapter,
        tokenizer=tokenizer,
        data_root=data_root,
        output=output,
        task=task,
        max_samples=max_samples,
        max_new_tokens=max_new_tokens,
    )
    atomic_write_json(output / "evaluation_summary.json", {"variant": variant, "tasks": summaries})
    typer.echo(f"evaluation written to {output}")


@app.command()
def latency(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/latency")),
    variant: str = typer.Option("full"),
    prompt: str = typer.Option("Explain dynamic semantic adaptation in one sentence."),
) -> None:
    """Run the paper's 10-warmup/1000-measurement synchronized latency protocol."""
    _empty_output(output)
    cfg = load_config(config)
    adapter, tokenizer = load_runtime(cfg)
    _install_checkpoint(adapter, checkpoint, cfg.anchor.residual_blend)
    configure_variant(adapter.controller, variant)
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)[-cfg.runtime.context_length :]
    result = benchmark_adapter(
        adapter,
        prompt_ids,
        warmup_steps=cfg.runtime.warmup_steps,
        measurement_steps=cfg.runtime.measurement_steps,
    )
    write_manifest(output, cfg, "measured")
    atomic_write_json(output / "latency.json", {"variant": variant, **result})
    typer.echo(f"latency result written to {output}")


@app.command()
def ablate(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    data_root: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/ablation")),
    task: str = typer.Option("hotpotqa"),
    max_samples: int | None = typer.Option(None, min=1),
) -> None:
    """Run full, no-approximation, no-pruning, and FP16-only variants."""
    _empty_output(output)
    cfg = load_config(config)
    adapter, tokenizer = load_runtime(cfg)
    _install_checkpoint(adapter, checkpoint, cfg.anchor.residual_blend)
    write_manifest(output, cfg, "measured")
    records = []
    for variant in ("full", "no-approximation", "no-pruning", "fp16-only"):
        configure_variant(adapter.controller, variant)
        variant_dir = output / variant
        variant_dir.mkdir()
        summaries = _evaluate_loaded(
            cfg=cfg,
            adapter=adapter,
            tokenizer=tokenizer,
            data_root=data_root,
            output=variant_dir,
            task=task,
            max_samples=max_samples,
            max_new_tokens=None,
        )
        records.append({"variant": variant, "tasks": summaries})
    atomic_write_json(output / "ablation.json", {"variants": records})
    typer.echo(f"ablation results written to {output}")


@app.command()
def sweep(
    checkpoint: Path = typer.Option(..., exists=True, dir_okay=False),
    data_root: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(Path("configs/paper_a100.yaml"), exists=True),
    output: Path = typer.Option(Path("artifacts/paper-a100/gate-sweep")),
    max_samples: int | None = typer.Option(None, min=1),
) -> None:
    """Re-run the HotpotQA gate-threshold sensitivity sweep."""
    _empty_output(output)
    cfg = load_config(config)
    adapter, tokenizer = load_runtime(cfg)
    _install_checkpoint(adapter, checkpoint, cfg.anchor.residual_blend)
    write_manifest(output, cfg, "measured")
    records = []
    for threshold in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
        adapter.controller.approximator.gate_threshold = threshold
        threshold_dir = output / f"gate-{threshold:.2f}"
        threshold_dir.mkdir()
        summary = _evaluate_loaded(
            cfg=cfg,
            adapter=adapter,
            tokenizer=tokenizer,
            data_root=data_root,
            output=threshold_dir,
            task="hotpotqa",
            max_samples=max_samples,
            max_new_tokens=None,
        )[0]
        records.append(
            {
                "gate_threshold": threshold,
                "approximation_ratio": summary["approximation_ratio"],
                "rouge_l": summary.get("metrics", {}).get("rouge_l", 0.0),
                "result_kind": "measured",
            }
        )
    atomic_write_json(output / "gate_sweep.json", {"points": records})
    with (output / "gate_sweep.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    typer.echo(f"gate sweep written to {output}")


@app.command()
def report(
    run_dir: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(Path("artifacts/report")),
    reference_dir: Path = typer.Option(Path("paper_reference"), exists=True),
) -> None:
    """Build a provenance-checked table/figure report."""
    result = build_report(run_dirs=[run_dir], reference_dir=reference_dir, output_dir=output)
    typer.echo(f"report written to {result.summary_markdown}")


@app.command("judge-results")
def judge_results(
    predictions: Path = typer.Option(..., exists=True, dir_okay=False),
    output: Path = typer.Option(Path("artifacts/judge")),
    config: Path = typer.Option(Path("configs/current_compatible.yaml"), exists=True),
    dry_run: bool = typer.Option(False),
) -> None:
    """Score AlpacaEval predictions with cached OpenAI Responses requests."""
    _empty_output(output)
    cfg = load_config(config)
    judge = JudgeClient(
        model=cfg.evaluation.judge_model,
        cache_dir=output,
        max_output_tokens=cfg.evaluation.judge_max_tokens,
        allow_missing_key_for_dry_run=dry_run,
    )
    results = []
    for row in read_jsonl(predictions):
        results.append(judge.score(row["prompt"], row["prediction"], dry_run=dry_run))
    atomic_write_json(output / "judge_results.json", {"results": results, "count": len(results)})
    typer.echo(f"judge output written to {output}")


if __name__ == "__main__":
    app()
