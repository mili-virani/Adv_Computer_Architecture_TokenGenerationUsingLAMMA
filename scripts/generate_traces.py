import os
import re
import time
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.profiler import profile, ProfilerActivity


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


@torch.no_grad()
def export_trace(model, tok, dev: torch.device, prompt_len: int, steps: int, out_path: str):
    input_ids = build_prompt(tok, prompt_len).to(dev)
    attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=dev)

    # tiny warmup outside profiler (important)
    sync(dev)
    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    sync(dev)

    activities = [ProfilerActivity.CPU]
    if dev.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    sync(dev)
    with profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        # Prefill
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        past = out.past_key_values
        next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

        # Decode
        for _ in range(steps):
            out = model(input_ids=next_token, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

    sync(dev)
    prof.export_chrome_trace(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--out_dir", default="results/traces")
    args = ap.parse_args()

    models = [
        "meta-llama/Llama-3.2-1B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
    ]
    prompt_lens = [128, 256, 512, 1024]

    dev = torch.device(args.device)
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    for model_id in models:
        print(f"\n=== Loading {model_id} on {dev.type} (requested {args.dtype}) ===")
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        try:
            model = model.to(dev, dtype=torch_dtype)
        except TypeError:
            model = model.to(dev)
        model.eval()

        actual_dtype = next(model.parameters()).dtype
        actual_dtype_str = str(actual_dtype).replace("torch.", "")
        print(f"[INFO] Actual dtype: {actual_dtype}")

        for L in prompt_lens:
            out_path = os.path.join(
                args.out_dir,
                f"trace_{safe_name(model_id)}_{dev.type}_dtype-{actual_dtype_str}_prompt{L}_steps{args.steps}.json",
            )
            print(f"Profiling prompt_len={L}, steps={args.steps} -> {out_path}")
            t0 = time.perf_counter()
            export_trace(model, tok, dev, L, args.steps, out_path)
            t1 = time.perf_counter()
            print(f"  Saved in {t1 - t0:.1f}s")

    print("\nDone. Traces are in results/traces/")
    print("Open with chrome://tracing or https://ui.perfetto.dev")


if __name__ == "__main__":
    main()