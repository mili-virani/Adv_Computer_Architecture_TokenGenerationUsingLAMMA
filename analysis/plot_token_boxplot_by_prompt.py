import os
import pandas as pd
import matplotlib.pyplot as plt

def main():
    results_dir = "results"
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)

    t1 = pd.read_csv(os.path.join(results_dir, "latency_tokens_meta-llama_Llama-3.2-1B-Instruct_mps.csv"))
    t3 = pd.read_csv(os.path.join(results_dir, "latency_tokens_meta-llama_Llama-3.2-3B-Instruct_mps.csv"))

    # steady-state only
    t1 = t1[t1["token_index"] > 0]
    t3 = t3[t3["token_index"] > 0]

    prompt_lens = sorted(set(t1["prompt_len"].unique()).union(set(t3["prompt_len"].unique())))

    data = []
    labels = []
    for L in prompt_lens:
        data.append(t1[t1["prompt_len"] == L]["token_ms"].values)
        labels.append(f"1B\n{L}")
        data.append(t3[t3["prompt_len"] == L]["token_ms"].values)
        labels.append(f"3B\n{L}")

    plt.figure(figsize=(12, 4))
    plt.boxplot(data, showfliers=False)
    plt.xticks(range(1, len(labels) + 1), labels)
    plt.ylabel("Per-token latency (ms)")
    plt.title("Per-token Latency Distribution by Prompt Length (steady-state tokens)")
    plt.grid(True, axis="y", alpha=0.3)

    out = os.path.join(out_dir, "per_token_boxplot_by_prompt.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print("Saved:", out)

if __name__ == "__main__":
    main()