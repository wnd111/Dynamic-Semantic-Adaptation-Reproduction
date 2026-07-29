#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Set HF_TOKEN before running this script}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before live judge evaluation}"

config_path="${DSA_CONFIG:-configs/paper_a100.yaml}"
run_path="${DSA_RUN_DIR:-artifacts/paper-a100}"
checkpoint_path="$run_path/checkpoints/auxiliary.pt"

dsa-repro preflight --config "$config_path" --output "$run_path/preflight.json"
for dataset_name in alpacaeval vicuna80 hotpotqa asqa sharegpt512; do
  dsa-repro prepare-data "$dataset_name" --output "$run_path/data"
done
dsa-repro calibrate --config "$config_path" --auxiliary-data "$run_path/data/sharegpt512/prepared.jsonl" --output "$run_path/calibration"
dsa-repro train --config "$config_path" --traces "$run_path/calibration" --output "$run_path/checkpoints"
dsa-repro evaluate --config "$config_path" --checkpoint "$checkpoint_path" --data-root "$run_path/data" --output "$run_path/evaluation"
dsa-repro latency --config "$config_path" --checkpoint "$checkpoint_path" --output "$run_path/latency"
dsa-repro ablate --config "$config_path" --checkpoint "$checkpoint_path" --data-root "$run_path/data" --output "$run_path/ablation"
dsa-repro sweep --config "$config_path" --checkpoint "$checkpoint_path" --data-root "$run_path/data" --output "$run_path/gate-sweep"
dsa-repro report --run-dir "$run_path/gate-sweep" --output "$run_path/report"
