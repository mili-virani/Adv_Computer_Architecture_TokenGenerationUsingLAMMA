import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--prompt_len", type=int, default=512)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    t1 = pd.read_csv(os.path.join(args.results_dir, "latency_tokens_meta-llama_Llama-3.2-1B-Instruct_mps.csv"))
    t3 = pd.read_csv(os.path.join(args.results_dir, "latency_tokens_meta-llama_Llama-3.2-3B-Instruct_mps.csv"))

    a = t1[t1["prompt_len"] == args.prompt_len].copy()
    b = t3[t3["prompt_len"] == args.prompt_len].copy()

    if a.empty or b.empty:
        raise RuntimeError(f"No token rows found for prompt_len={args.prompt_len}")

    a_mean = a.groupby("token_index")["token_ms"].mean().reset_index()
    b_mean = b.groupby("token_index")["token_ms"].mean().reset_index()

    plt.figure()
    plt.plot(a_mean["token_index"], a_mean["token_ms"], label="Llama-3.2-1B")
    plt.plot(b_mean["token_index"], b_mean["token_ms"], label="Llama-3.2-3B")
    plt.xlabel("Token index")
    plt.ylabel("Latency per token (ms)")
    plt.title(f"Mean Per-token Latency vs Token Index (prompt_len={args.prompt_len})")
    plt.grid(True, alpha=0.3)
    plt.legend()

    out = os.path.join(args.out_dir, f"token_latency_vs_index_prompt{args.prompt_len}.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()