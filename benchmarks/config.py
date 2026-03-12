"""
benchmarks.config

Central configuration for Project 6 latency benchmarks.
Used by benchmarks.scaling to run reproducible sweeps.
"""

# -------------------------------------------------
# Models (two sizes required by the assignment)
# -------------------------------------------------
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
]

# -------------------------------------------------
# Required constraint
# -------------------------------------------------
BATCH_SIZE = 1

# -------------------------------------------------
# Prompt length sweep (for scaling plots)
# -------------------------------------------------
PROMPT_LENGTHS = [128, 256, 512, 1024]

# -------------------------------------------------
# Fixed output length
# -------------------------------------------------
OUTPUT_LENGTH = 128

# -------------------------------------------------
# Warmups + trials (required by rubric)
# -------------------------------------------------
WARMUP_RUNS = 3
TRIALS = 10

# -------------------------------------------------
# Outlier handling
# -------------------------------------------------
TRIM_RATIO = 0.10

# -------------------------------------------------
# Profiling breakdown defaults
# -------------------------------------------------
PROFILE_PROMPT_TOKENS = 256
PROFILE_STEPS = 50