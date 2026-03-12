import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"   # change to 1B if you want
PROMPT = "Hello! Explain what token generation is in one sentence."
NEW_TOKENS = 10

print("Torch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# IMPORTANT: force dtype to float16 on MPS (faster + lower memory)
dtype = torch.float16 if device.type == "mps" else torch.float32
print("Requested dtype:", dtype)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    low_cpu_mem_usage=True,
).to(device)

# Confirm model device + dtype
p = next(model.parameters())
print("Model param device:", p.device)
print("Model param dtype:", p.dtype)

inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

# Warmup
with torch.no_grad():
    _ = model.generate(**inputs, max_new_tokens=2)

# Timed generation
start = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=NEW_TOKENS)
end = time.time()

text = tokenizer.decode(out[0], skip_special_tokens=True)
print("\nGenerated text:\n", text[:200], "...\n")

total_s = end - start
print(f"Total time for {NEW_TOKENS} new tokens: {total_s:.3f}s")
print(f"Approx time/token: {(total_s/NEW_TOKENS)*1000:.1f} ms/token")

# MPS memory (if supported)
if device.type == "mps":
    try:
        print("MPS current allocated (bytes):", torch.mps.current_allocated_memory())
        print("MPS driver allocated (bytes):", torch.mps.driver_allocated_memory())
    except Exception as e:
        print("MPS memory query not available:", e)