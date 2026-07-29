#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment-a100.yml
conda run -n dsa-repro-a100 python -m pip install -e .
conda run -n dsa-repro-a100 python -m pytest -q
