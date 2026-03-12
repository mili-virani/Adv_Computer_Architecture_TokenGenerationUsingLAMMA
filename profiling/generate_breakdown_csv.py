import argparse
import os
import re
from collections import defaultdict

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import AutoModelForCausalLM, AutoTokenizer


def safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", s)


def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def build_prompt(tokenizer, prompt_len: int) -> torch.Tensor:
    seed = (
        "The transformer architecture has revolutionized natural language processing. "
        "Autoregressive decoding makes per-token latency critical. KV-cache grows with "
        "sequence length and can become bandwidth-bound. "
    ) * 80
    ids = tokenizer(seed, return_tensors="pt").input_ids[0].tolist()
    prompt_ids = (ids * ((prompt_len // len(ids)) + 1))[:prompt_len]
    return torch.tensor([prompt_ids], dtype=torch.long)

def bucket_for(name: str) -> str:
    n = name.lower()

    # --- Embedding / token lookup ---
    if any(k in n for k in [
        "embedding", "embed", "token_embedding", "index_select", "gather"
    ]):
        return "embedding"

    # --- LayerNorm / RMSNorm / residual ops ---
    if any(k in n for k in [
        "layernorm", "rmsnorm", "native_layer_norm", "batch_norm", "norm",
        "add_", "add ", "residual", "dropout"
    ]):
        return "layernorm_residual"

    # --- Attention / KV / Softmax ---
    if any(k in n for k in [
        "attention", "attn", "scaled_dot_product", "sdpa", "softmax",
        "qkv", "kv", "k_cache", "v_cache", "transpose", "permute",
        "matmul", "bmm", "mm"
    ]):
        return "attention"

    # --- MLP / FFN ---
    if any(k in n for k in [
        "mlp", "ffn", "linear", "addmm", "gemm",
        "silu", "gelu", "relu", "mul", "div"
    ]):
        return "mlp"

    # --- Sampling / decoding ---
    if any(k in n for k in [
        "argmax", "topk", "multinomial"
    ]):
        return "sampling"

    # --- Framework / overhead ---
    if any(k in n for k in [
        "copy", "to(", "empty", "resize", "contiguous", "cat",
        "view", "reshape", "slice", "select", "index",
        "synchronize", "wait", "dispatch", "compile"
    ]):
        return "framework_overhead"

    return "other"


@torch.no_grad()
def run_and_collect_breakdown(model, tok, dev: torch.device, prompt_len: int, steps: int):
    input_ids = build_prompt(tok, prompt_len).to(dev)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)

    # small warmup (not profiled)
    sync(dev)
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    sync(dev)

    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    sync(dev)
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        # Prefill
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # Decode loop
        for _ in range(steps):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    sync(dev)

    # Aggregate operator time
    # CPU times are in us; CUDA times are also in us (if present)
    buckets_cpu_us = defaultdict(float)
    buckets_cuda_us = defaultdict(float)

    for evt in prof.key_averages():
        name = evt.key
        b = bucket_for(name)

        buckets_cpu_us[b] += float(evt.self_cpu_time_total)

        # cuda may be missing on mps/cpu
        cuda_us = getattr(evt, "self_cuda_time_total", 0.0)
        try:
            buckets_cuda_us[b] += float(cuda_us)
        except Exception:
            pass

    return buckets_cpu_us, buckets_cuda_us


def write_breakdown_csv(out_path: str, buckets_cpu_us, buckets_cuda_us):
    import pandas as pd

    rows = []
    all_keys = sorted(set(buckets_cpu_us.keys()) | set(buckets_cuda_us.keys()))
    for k in all_keys:
        cpu_ms = buckets_cpu_us.get(k, 0.0) / 1000.0
        cuda_ms = buckets_cuda_us.get(k, 0.0) / 1000.0
        rows.append({"bucket": k, "cpu_ms": cpu_ms, "cuda_ms": cuda_ms})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Saved breakdown CSV:", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--prompt_len", type=int, default=512)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    dev = torch.device(args.device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    try:
        model = model.to(dev, dtype=torch_dtype)
    except TypeError:
        model = model.to(dev)
    model.eval()

    cpu_us, cuda_us = run_and_collect_breakdown(model, tok, dev, args.prompt_len, args.steps)

    out_path = os.path.join(
        args.out_dir,
        f"breakdown_{safe_name(args.model)}_prompt{args.prompt_len}_steps{args.steps}.csv",
    )
    write_breakdown_csv(out_path, cpu_us, cuda_us)


if __name__ == "__main__":
    main()