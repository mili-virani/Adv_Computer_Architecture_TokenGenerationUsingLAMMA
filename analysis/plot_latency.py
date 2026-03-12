import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_two(df_a, df_b, x, y, label_a, label_b, title, out_path, xlabel, ylabel):
    plt.figure()
    plt.plot(df_a[x], df_a[y], marker="o", label=label_a)
    plt.plot(df_b[x], df_b[y], marker="o", label=label_b)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="plots")
    ap.add_argument("--model_a_tag", default="meta-llama_Llama-3.2-1B-Instruct")
    ap.add_argument("--model_b_tag", default="meta-llama_Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # find summary csvs
    a_files = glob.glob(os.path.join(args.results_dir, f"latency_summary_*{args.model_a_tag}*_*.csv"))
    b_files = glob.glob(os.path.join(args.results_dir, f"latency_summary_*{args.model_b_tag}*_*.csv"))
    if not a_files or not b_files:
        raise RuntimeError("Could not find summary CSVs. Check results_dir and filenames.")

    df_a = pd.read_csv(sorted(a_files)[-1]).sort_values("prompt_len")
    df_b = pd.read_csv(sorted(b_files)[-1]).sort_values("prompt_len")

    plot_two(df_a, df_b, "prompt_len", "ttft_trim_ms",
             "Llama-3.2-1B", "Llama-3.2-3B",
             "TTFT vs Prompt Length (Batch=1)",
             os.path.join(args.out_dir, "ttft_vs_prompt.png"),
             "Prompt length (tokens)", "TTFT (ms, trimmed mean)")

    plot_two(df_a, df_b, "prompt_len", "per_token_trim_ms",
             "Llama-3.2-1B", "Llama-3.2-3B",
             "Per-token Decode Latency vs Prompt Length (Batch=1)",
             os.path.join(args.out_dir, "per_token_vs_prompt.png"),
             "Prompt length (tokens)", "Per-token latency (ms, trimmed mean)")

    plot_two(df_a, df_b, "prompt_len", "e2e_trim_ms",
             "Llama-3.2-1B", "Llama-3.2-3B",
             "End-to-end Latency vs Prompt Length (Batch=1)",
             os.path.join(args.out_dir, "e2e_vs_prompt.png"),
             "Prompt length (tokens)", "End-to-end latency (ms, trimmed mean)")

    print("Saved latency plots to:", args.out_dir)

if __name__ == "__main__":
    main()