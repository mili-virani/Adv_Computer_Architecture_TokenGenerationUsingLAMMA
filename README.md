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
hf auth login
```

### Windows

```bat
.\scripts\setup_windows.bat
.venv\Scripts\activate
hf auth login
```

## Running (Examples)

### Run benchmark

```bash
python -m src.benchmark --model meta-llama/Llama-3.2-1B-Instruct --prompt-length 128 --output-tokens 128 --batch-size 1 --trials 10 --warmups 3 --device mps --dtype fp16 --out results/llama32_1b_L128.csv
```

### Run benchmark sweep using multiple models and prompt lengths

```bash
python -m src.benchmark_sweep --models meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct --prompt-lengths 128 256 512 1024 --output-tokens 128 --batch-size 1 --trials 10 --warmups 3 --device mps --out results/scaling.csv
```

### Measure component-level latency breakdown

```bash
python -m src.decompose_latency --model meta-llama/Llama-3.2-1B-Instruct --prompt-length 512 --output-tokens 128 --batch-size 1 --device mps --out results/decomposition_1b_L512.json
```

### Plot benchmark data

```bash
python -m scripts.plot_results --input results/scaling.csv --out-dir figures
```

## Third-Party Libraries

This repo uses the following third-party libraries:

- From [Hugging Face](https://huggingface.co):
    - [Accelerate](https://huggingface.co/docs/accelerate/index)
    - [Hub client library](https://huggingface.co/docs/huggingface_hub)
    - [Tokenizers](https://huggingface.co/docs/tokenizers/en/index)
    - [Transformers](https://huggingface.co/docs/transformers/index)
- [Matplotlib](https://matplotlib.org)
- [NumPy](https://numpy.org)
- [pandas](https://pandas.pydata.org)
- [PyTorch](https://pytorch.org)
- [SentencePiece](https://github.com/google/sentencepiece)
- [tqdm](https://github.com/tqdm/tqdm)
