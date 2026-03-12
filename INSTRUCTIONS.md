# Project 6: Token-Generation Latency Benchmarking in LLaMA (Advanced)

This repo benchmarks token-generation latency for LLaMA-style models with:
- TTFT (time-to-first-token)
- Per-token decode latency (token-by-token)
- End-to-end latency
- Latency decomposition (Embedding / Attention / MLP / Norm / Sampling / Framework)
- Scaling analysis over prompt length and model size (1B vs 3B)
- Profiler trace export for architectural reasoning

## Setup
### Linux/macOS
```bash
./scripts/setup_env.sh
source .venv/bin/activate
huggingface-cli login