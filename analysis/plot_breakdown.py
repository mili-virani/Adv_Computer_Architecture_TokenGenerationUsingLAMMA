import argparse
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_breakdown(csv_path: str, out_path: str):
    df = pd.read_csv(csv_path)
    use_cuda = ("cuda_ms" in df.columns) and (df["cuda_ms"].sum() > 0)

    if use_cuda:
        df = df.sort_values("cuda_ms", ascending=False)
        vals = df["cuda_ms"]
        ylabel = "CUDA time (ms)"
        title = "Latency Breakdown (CUDA time)"
    else:
        df = df.sort_values("cpu_ms", ascending=False)
        vals = df["cpu_ms"]
        ylabel = "CPU time (ms)"
        title = "Latency Breakdown (CPU time)"

    plt.figure(figsize=(9, 4))
    plt.bar(df["bucket"], vals)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.grid(True, axis="y", alpha=0.3)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="plots")

    # ✅ IMPORTANT: your filenames contain "meta-llama_" prefix, so use that in tags
    ap.add_argument("--model_a_tag", default="meta-llama_Llama-3.2-1B-Instruct")
    ap.add_argument("--model_b_tag", default="meta-llama_Llama-3.2-3B-Instruct")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    a_files = glob.glob(os.path.join(args.results_dir, f"breakdown_*{args.model_a_tag}*_prompt*_steps*.csv"))
    b_files = glob.glob(os.path.join(args.results_dir, f"breakdown_*{args.model_b_tag}*_prompt*_steps*.csv"))

    # ✅ If you have neither, profiling truly hasn't run
    if not a_files and not b_files:
        raise RuntimeError("Could not find any breakdown CSVs. Run profiling first.")

    # ✅ Plot whatever exists (1B only now, later 1B + 3B)
    if a_files:
        a_csv = sorted(a_files)[-1]
        plot_breakdown(a_csv, os.path.join(args.out_dir, "breakdown_llama32_1b.png"))
        print("Saved 1B breakdown:", os.path.join(args.out_dir, "breakdown_llama32_1b.png"))
    else:
        print("No 1B breakdown CSV found. (Run profiling for 1B)")

    if b_files:
        b_csv = sorted(b_files)[-1]
        plot_breakdown(b_csv, os.path.join(args.out_dir, "breakdown_llama32_3b.png"))
        print("Saved 3B breakdown:", os.path.join(args.out_dir, "breakdown_llama32_3b.png"))
    else:
        print("No 3B breakdown CSV found yet (this is OK). Run profiling for 3B when ready.")

    print("Done. Breakdown plots in:", args.out_dir)


if __name__ == "__main__":
    main()