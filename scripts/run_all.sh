#!/usr/bin/env bash
set -euo pipefail

if [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
fi

mkdir -p results plots

# 1) Benchmark (Goal 1 + Goal 3: model size + prompt scaling)
python -m benchmarks.harness_hf \
  --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
  --prompt_lens 128 256 512 1024 \
  --gen_tokens 128 \
  --warmup 3 \
  --trials 10 \
  --device auto \
  --dtype fp16 \
  --out_dir results

# 2) Profiling breakdown (Goal 2)
python -m profiling.profile_breakdown_torch \
  --model meta-llama/Llama-3.2-1B-Instruct \
  --prompt_tokens 256 \
  --steps 50 \
  --device auto \
  --dtype fp16 \
  --out_dir results

python -m profiling.profile_breakdown_torch \
  --model meta-llama/Llama-3.2-3B-Instruct \
  --prompt_tokens 256 \
  --steps 50 \
  --device auto \
  --dtype fp16 \
  --out_dir results

# 3) Plots (Goal 3)
python -m analysis.plot_latency --results_dir results --out_dir plots
python -m analysis.plot_breakdown --results_dir results --out_dir plots

echo "Done. Outputs in results/ and plots/."