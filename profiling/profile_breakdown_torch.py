import argparse
import os
from collections import defaultdict
from typing import Dict

import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import AutoModelForCausalLM, AutoTokenizer

from benchmarks.utils import safe_name, write_csv

def pick_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def sync(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.synchronize()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()

def dtype_from_str(s: str):
    if s == "fp16":
        return torch.float16
    if s == "bf16":
        return torch.bfloat16
    if s == "fp32":
        return torch.float32
    return None

def build_prompt(tokenizer, prompt_len: int) -> torch.Tensor:
    seed = (
        "Transformers use self-attention. During autoregressive decoding, KV-cache grows with sequence length. "
        "Latency per generated token is critical. "
    ) * 120
    ids = tokenizer(seed, return_tensors="pt").input_ids[0].tolist()
    prompt_ids = (ids * ((prompt_len // len(ids)) + 1))[:prompt_len]
    return torch.tensor([prompt_ids], dtype=torch.long)

def bucket_for_name(module_name: str) -> str:
    n = module_name.lower()
    if "embed_tokens" in n or "embedding" in n:
        return "Embedding"
    if "self_attn" in n or "attention" in n or ".attn" in n:
        return "Attention"
    if ".mlp" in n or "mlp" in n:
        return "MLP"
    if "layernorm" in n or "input_layernorm" in n or "post_attention_layernorm" in n or ".norm" in n:
        return "LayerNorm/Residual"
    return "OtherModelOps"

def attach_bucket_hooks(model):
    """
    Wrap key submodules in record_function("BUCKET::<name>") for torch.profiler attribution.
    """
    handles = []
    active = {}

    def pre(mod, inp, name: str):
        b = bucket_for_name(name)
        ctx = record_function(f"BUCKET::{b}")
        active[id(mod)] = ctx
        ctx.__enter__()

    def post(mod, inp, out):
        ctx = active.pop(id(mod), None)
        if ctx is not None:
            ctx.__exit__(None, None, None)

    for name, mod in model.named_modules():
        ln = name.lower()
        if any(k in ln for k in ["embed_tokens", "self_attn", "mlp", "layernorm", "input_layernorm", "post_attention_layernorm", ".norm"]):
            handles.append(mod.register_forward_pre_hook(lambda m, i, nn=name: pre(m, i, nn)))
            handles.append(mod.register_forward_hook(post))

    return handles

@torch.no_grad()
def run_profile(model, input_ids, attention_mask, steps: int):
    # Prefill
    with record_function("PHASE::PREFILL"):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values

    next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    for _ in range(steps):
        with record_function("PHASE::DECODE_STEP"):
            out = model(input_ids=next_tok, past_key_values=past, use_cache=True)
            past = out.past_key_values
            with record_function("BUCKET::Sampling"):
                next_tok = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

def summarize(prof) -> Dict[str, Dict[str, float]]:
    buckets = defaultdict(lambda: {"cpu_ms": 0.0, "cuda_ms": 0.0})
    for evt in prof.key_averages():
        k = evt.key
        if not k.startswith("BUCKET::"):
            continue
        b = k.split("BUCKET::", 1)[1]
        buckets[b]["cpu_ms"] += evt.cpu_time_total / 1000.0
        if hasattr(evt, "cuda_time_total"):
            buckets[b]["cuda_ms"] += evt.cuda_time_total / 1000.0
    return buckets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--prompt_tokens", type=int, default=256)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    dev = pick_device(args.device)
    torch_dtype = dtype_from_str(args.dtype)
    if dev.type == "cpu" and torch_dtype == torch.float16:
        torch_dtype = None

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map=None,
        low_cpu_mem_usage=True,
    )
    if torch_dtype is not None:
        try:
            model = model.to(dev, dtype=torch_dtype)
        except TypeError:
            model = model.to(dev)
    else:
        model = model.to(dev)
    model.eval()

    input_ids = build_prompt(tok, args.prompt_tokens).to(dev)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)

    # Warmup a bit
    for _ in range(2):
        run_profile(model, input_ids, attention_mask, steps=min(5, args.steps))
    sync(dev)

    handles = attach_bucket_hooks(model)

    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    safe = safe_name(args.model)
    os.makedirs(args.out_dir, exist_ok=True)

    trace_path = os.path.join(args.out_dir, f"trace_{safe}_{dev.type}_prompt{args.prompt_tokens}_steps{args.steps}.json")
    breakdown_path = os.path.join(args.out_dir, f"breakdown_{safe}_{dev.type}_prompt{args.prompt_tokens}_steps{args.steps}.csv")

    with profile(activities=activities, profile_memory=True, record_shapes=False, with_stack=False) as prof:
        with record_function("PHASE::FULL_RUN"):
            run_profile(model, input_ids, attention_mask, steps=args.steps)

    # remove hooks
    for h in handles:
        h.remove()

    prof.export_chrome_trace(trace_path)

    buckets = summarize(prof)

    # compute totals for percent shares
    total_cpu = sum(v["cpu_ms"] for v in buckets.values())
    total_cuda = sum(v["cuda_ms"] for v in buckets.values())

    rows = []
    for b, t in buckets.items():
        rows.append({
            "model": args.model,
            "device": dev.type,
            "dtype": args.dtype,
            "prompt_tokens": args.prompt_tokens,
            "steps": args.steps,
            "bucket": b,
            "cpu_ms": t["cpu_ms"],
            "cuda_ms": t["cuda_ms"],
            "cpu_pct": (t["cpu_ms"] / total_cpu * 100.0) if total_cpu > 0 else 0.0,
            "cuda_pct": (t["cuda_ms"] / total_cuda * 100.0) if total_cuda > 0 else 0.0,
        })

    rows.sort(key=lambda r: r["cuda_ms"], reverse=True)
    write_csv(breakdown_path, rows)

    print(f"Saved breakdown: {breakdown_path}")
    print(f"Saved trace: {trace_path}")

if __name__ == "__main__":
    main()