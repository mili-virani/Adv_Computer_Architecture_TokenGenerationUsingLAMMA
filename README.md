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
```

### Windows

```bat
.\scripts\setup_windows.bat
.venv\Scripts\activate
huggingface-cli login
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
