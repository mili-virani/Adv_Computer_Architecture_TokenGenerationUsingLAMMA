import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--dtype", choices=["fp16","bf16","fp32","auto"], default="auto")
    args = ap.parse_args()

    for m in args.models:
        print(f"Downloading tokenizer: {m}")
        AutoTokenizer.from_pretrained(m, use_fast=True)

        print(f"Downloading model: {m}")
        if args.dtype == "fp16" and torch.cuda.is_available():
            AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        elif args.dtype == "bf16" and torch.cuda.is_available():
            AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
        else:
            AutoModelForCausalLM.from_pretrained(m, low_cpu_mem_usage=True)

        print(f"Cached: {m}\n")

if __name__ == "__main__":
    main()