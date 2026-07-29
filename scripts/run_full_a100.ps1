$ErrorActionPreference = "Stop"

if (-not $env:HF_TOKEN) { throw "Set HF_TOKEN before running this script" }
if (-not $env:OPENAI_API_KEY) { throw "Set OPENAI_API_KEY before live judge evaluation" }

$configPath = if ($env:DSA_CONFIG) { $env:DSA_CONFIG } else { "configs/paper_a100.yaml" }
$runPath = if ($env:DSA_RUN_DIR) { $env:DSA_RUN_DIR } else { "artifacts/paper-a100" }
$checkpointPath = Join-Path $runPath "checkpoints/auxiliary.pt"

dsa-repro preflight --config $configPath --output (Join-Path $runPath "preflight.json")
foreach ($datasetName in @("alpacaeval", "vicuna80", "hotpotqa", "asqa", "sharegpt512")) {
    dsa-repro prepare-data $datasetName --output (Join-Path $runPath "data")
}
dsa-repro calibrate --config $configPath --auxiliary-data (Join-Path $runPath "data/sharegpt512/prepared.jsonl") --output (Join-Path $runPath "calibration")
dsa-repro train --config $configPath --traces (Join-Path $runPath "calibration") --output (Join-Path $runPath "checkpoints")
dsa-repro evaluate --config $configPath --checkpoint $checkpointPath --data-root (Join-Path $runPath "data") --output (Join-Path $runPath "evaluation")
dsa-repro latency --config $configPath --checkpoint $checkpointPath --output (Join-Path $runPath "latency")
dsa-repro ablate --config $configPath --checkpoint $checkpointPath --data-root (Join-Path $runPath "data") --output (Join-Path $runPath "ablation")
dsa-repro sweep --config $configPath --checkpoint $checkpointPath --data-root (Join-Path $runPath "data") --output (Join-Path $runPath "gate-sweep")
dsa-repro report --run-dir (Join-Path $runPath "gate-sweep") --output (Join-Path $runPath "report")
