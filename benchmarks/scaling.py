from benchmarks.config import MODELS, PROMPT_LENGTHS, OUTPUT_LENGTH, WARMUP_RUNS, TRIALS, TRIM_RATIO
import subprocess
import sys

def main():
    cmd = [
        sys.executable, "-m", "benchmarks.harness_hf",
        "--models", *MODELS,
        "--prompt_lens", *[str(x) for x in PROMPT_LENGTHS],
        "--gen_tokens", str(OUTPUT_LENGTH),
        "--warmup", str(WARMUP_RUNS),
        "--trials", str(TRIALS),
        "--trim", str(TRIM_RATIO),
        "--device", "auto",
        "--dtype", "fp16",
        "--out_dir", "results",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    main()