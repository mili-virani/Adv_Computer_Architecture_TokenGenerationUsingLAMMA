import argparse
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.utils import (
    safe_name, write_csv, trimmed_mean, iqr_filter, stats_dict
)

@dataclass
class SummaryRow:
    model: str
    device: str
    dtype: str
    prompt_len: int
    gen_tokens: int
    trials: int
    warmup: int
    ttft_trim_ms: float
    ttft_p50_ms: float
    ttft_p90_ms: float
    ttft_p99_ms: float
    per_token_trim_ms: float
    per_token_p50_ms: float
    per_token_p90_ms: float
    per_token_p99_ms: float
    e2e_trim_ms: float
    e2e_p50_ms: float
    e2e_p90_ms: float
    e2e_p99_ms: float
    peak_mem_mb_mean: float

def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync(dev: torch.device) -> None:
    """Synchronize the current device to ensure wall-clock timings are accurate."""
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        # MPS is asynchronous; synchronize for correct wall-clock timings.
        torch.mps.synchronize()

def dtype_from_str(s: str):
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    return None  # auto

def build_prompt(tokenizer, prompt_len: int) -> torch.Tensor:
    seed = (
        "The transformer architecture has revolutionized natural language processing. "
        "Autoregressive decoding makes per-token latency critical. KV-cache grows with "
        "sequence length and can become bandwidth-bound. "
    ) * 80
    ids = tokenizer(seed, return_tensors="pt").input_ids[0].tolist()
    if len(ids) < 32:
        raise RuntimeError("Seed prompt too short; increase seed text.")
    prompt_ids = (ids * ((prompt_len // len(ids)) + 1))[:prompt_len]
    return torch.tensor([prompt_ids], dtype=torch.long)

@torch.no_grad()
def prefill_and_decode(
    model,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    gen_tokens: int,
    dev: torch.device,
) -> Tuple[float, List[float], float]:
    """
    Returns:
      ttft_ms = prefill + first decode step
      per_token_ms = per token decode step latency list (len == gen_tokens)
      e2e_ms = prefill + sum(per_token_ms)
    """
    assert input_ids.shape[0] == 1, "Batch size must be 1."

    use_cuda_events = (dev.type == "cuda")

    # -------- Prefill --------
    # Sync BEFORE starting timing to avoid queued work bleeding into this region.
    sync(dev)

    if use_cuda_events:
        pre_s = torch.cuda.Event(enable_timing=True)
        pre_e = torch.cuda.Event(enable_timing=True)
        pre_s.record()
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        pre_e.record()
        sync(dev)
        prefill_ms = float(pre_s.elapsed_time(pre_e))
    else:
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        # Sync AFTER to ensure the work has finished before stopping the timer.
        sync(dev)
        prefill_ms = (time.perf_counter() - t0) * 1000.0

    past = out.past_key_values
    next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    # -------- Decode loop (true per-token) --------
    per_token_ms: List[float] = []
    for _ in range(gen_tokens):
        if use_cuda_events:
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()

            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            e.record()
            sync(dev)
            step_ms = float(s.elapsed_time(e))
        else:
            # ✅ IMPORTANT: sync BEFORE and AFTER the timed region on async backends (MPS)
            sync(dev)
            t1 = time.perf_counter()

            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

            sync(dev)
            step_ms = (time.perf_counter() - t1) * 1000.0

        per_token_ms.append(step_ms)

    ttft_ms = prefill_ms + (per_token_ms[0] if per_token_ms else 0.0)
    e2e_ms = prefill_ms + sum(per_token_ms)
    return ttft_ms, per_token_ms, e2e_ms

def load_model_tokenizer(model_id: str, dev: torch.device, dtype_str: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch_dtype = dtype_from_str(dtype_str)

    # CPU fp16 is often unsupported/slow; keep CPU in fp32 unless explicitly using bf16.
    if dev.type == "cpu" and torch_dtype == torch.float16:
        torch_dtype = None

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # Use the requested dtype whenever possible to avoid silently benchmarking fp32.
        torch_dtype=torch_dtype,
        # Avoid sharding / offload for latency benchmarking unless explicitly intended.
        device_map=None,
        low_cpu_mem_usage=True,
    )

    # Move to device and (if supported) cast to requested dtype.
    if torch_dtype is not None:
        try:
            model = model.to(dev, dtype=torch_dtype)
        except TypeError:
            model = model.to(dev)
    else:
        model = model.to(dev)

    model.eval()

    # Record actual dtype (do not trust requested dtype on all backends)
    actual_dtype = next(model.parameters()).dtype
    print(f"[INFO] Actual model dtype on {dev.type}: {actual_dtype}")

    return model, tok, actual_dtype

def run_one_model(
    model_id: str,
    dev: torch.device,
    dtype_str: str,
    prompt_lens: List[int],
    gen_tokens: int,
    warmup: int,
    trials: int,
    trim_ratio: float,
    out_dir: str,
):
    safe = safe_name(model_id)
    summary_path = f"{out_dir}/latency_summary_{safe}_{dev.type}.csv"
    tokens_path = f"{out_dir}/latency_tokens_{safe}_{dev.type}.csv"

    model, tok, actual_dtype = load_model_tokenizer(model_id, dev, dtype_str)
    actual_dtype_str = str(actual_dtype).replace("torch.", "")

    summary_rows: List[Dict] = []
    token_rows: List[Dict] = []

    for prompt_len in prompt_lens:
        input_ids = build_prompt(tok, prompt_len).to(dev)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)

        # Warmups
        for _ in range(warmup):
            _ = prefill_and_decode(
                model, input_ids, attention_mask,
                gen_tokens=min(8, gen_tokens),
                dev=dev
            )

        ttfts, per_means, e2es, peaks = [], [], [], []
        for t in range(trials):
            if dev.type == "cuda":
                torch.cuda.reset_peak_memory_stats()

            ttft_ms, per_token_ms, e2e_ms = prefill_and_decode(
                model, input_ids, attention_mask,
                gen_tokens=gen_tokens,
                dev=dev
            )

            peak_mb = 0.0
            if dev.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            ttfts.append(ttft_ms)
            # Steady-state per-token latency excludes the first decode step.
            steady = per_token_ms[1:] if len(per_token_ms) > 1 else per_token_ms
            per_means.append(sum(steady) / len(steady) if steady else 0.0)
            e2es.append(e2e_ms)
            peaks.append(peak_mb)

            for i, ms in enumerate(per_token_ms):
                token_rows.append({
                    "model": model_id,
                    "device": dev.type,
                    "dtype": actual_dtype_str,   # ✅ actual dtype
                    "prompt_len": prompt_len,
                    "trial": t,
                    "token_index": i,
                    "token_ms": ms,
                })

        # Outlier filter (IQR) then compute robust stats
        ttft_f = iqr_filter(ttfts)
        per_f = iqr_filter(per_means)
        e2e_f = iqr_filter(e2es)

        ttft_stats = stats_dict(ttft_f)
        per_stats = stats_dict(per_f)
        e2e_stats = stats_dict(e2e_f)

        row = SummaryRow(
            model=model_id,
            device=dev.type,
            dtype=actual_dtype_str,  # ✅ actual dtype
            prompt_len=prompt_len,
            gen_tokens=gen_tokens,
            trials=trials,
            warmup=warmup,
            ttft_trim_ms=trimmed_mean(ttft_f, trim_ratio),
            ttft_p50_ms=ttft_stats["p50"],
            ttft_p90_ms=ttft_stats["p90"],
            ttft_p99_ms=ttft_stats["p99"],
            per_token_trim_ms=trimmed_mean(per_f, trim_ratio),
            per_token_p50_ms=per_stats["p50"],
            per_token_p90_ms=per_stats["p90"],
            per_token_p99_ms=per_stats["p99"],
            e2e_trim_ms=trimmed_mean(e2e_f, trim_ratio),
            e2e_p50_ms=e2e_stats["p50"],
            e2e_p90_ms=e2e_stats["p90"],
            e2e_p99_ms=e2e_stats["p99"],
            peak_mem_mb_mean=float(sum(peaks) / len(peaks)) if peaks else 0.0,
        )
        summary_rows.append(asdict(row))

        print(
            f"[{model_id}] prompt={prompt_len} "
            f"TTFT(trim)={row.ttft_trim_ms:.2f}ms  "
            f"PerTok(trim)={row.per_token_trim_ms:.2f}ms  "
            f"E2E(trim)={row.e2e_trim_ms:.2f}ms  "
            f"peak_mem~{row.peak_mem_mb_mean:.1f}MB"
        )

    write_csv(summary_path, summary_rows)
    write_csv(tokens_path, token_rows)
    print(f"Saved: {summary_path}")
    print(f"Saved: {tokens_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--prompt_lens", nargs="+", type=int, required=True)
    ap.add_argument("--gen_tokens", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--trim", type=float, default=0.10)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    dev = pick_device(args.device)
    for m in args.models:
        run_one_model(
            model_id=m,
            dev=dev,
            dtype_str=args.dtype,
            prompt_lens=args.prompt_lens,
            gen_tokens=args.gen_tokens,
            warmup=args.warmup,
            trials=args.trials,
            trim_ratio=args.trim,
            out_dir=args.out_dir,
        )

if __name__ == "__main__":
    main()