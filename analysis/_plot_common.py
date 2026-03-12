import os
import pandas as pd

SUMMARY_1B = "results/latency_summary_meta-llama_Llama-3.2-1B-Instruct_mps.csv"
SUMMARY_3B = "results/latency_summary_meta-llama_Llama-3.2-3B-Instruct_mps.csv"
TOKENS_1B  = "results/latency_tokens_meta-llama_Llama-3.2-1B-Instruct_mps.csv"
TOKENS_3B  = "results/latency_tokens_meta-llama_Llama-3.2-3B-Instruct_mps.csv"

def ensure_plots_dir():
    os.makedirs("plots", exist_ok=True)

def load_summaries():
    df1 = pd.read_csv(SUMMARY_1B).sort_values("prompt_len")
    df3 = pd.read_csv(SUMMARY_3B).sort_values("prompt_len")
    return df1, df3

def load_tokens():
    t1 = pd.read_csv(TOKENS_1B)
    t3 = pd.read_csv(TOKENS_3B)
    return t1, t3