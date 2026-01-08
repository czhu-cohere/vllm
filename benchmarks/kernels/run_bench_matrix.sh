#!/usr/bin/env bash
set -euo pipefail

MODEL="/root/engines/c5-8a133t-bf16-dummy-shallow/"

TPS=(2 4 8)

BASE_CMD=(python3 benchmark_moe.py --model "$MODEL" --tune)

for tp in "${TPS[@]}"; do
  echo "=============================="
  echo "Running: tp=${tp} (no expert parallel)"
  echo "=============================="
  "${BASE_CMD[@]}" -tp "$tp"

  echo "=============================="
  echo "Running: tp=${tp} (with --enable-expert-parallel)"
  echo "=============================="
  "${BASE_CMD[@]}" -tp "$tp" --enable-expert-parallel
done

echo "All runs completed."
