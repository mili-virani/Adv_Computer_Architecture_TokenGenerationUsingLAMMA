#!/usr/bin/env bash
set -euo pipefail

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p results plots

python -m benchmarks.harness_hf \
  --models meta-llama/Llama-3.2-1B-Instruct \
  --prompt_lens 128 \
  --gen_tokens 8 \
  --warmup 1 \
  --trials 1 \
  --device auto \
  --dtype fp16 \
  --out_dir results

echo "Smoke test done. Check results/."