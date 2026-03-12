import torch

print("Torch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
print("MPS built:", torch.backends.mps.is_built())

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Selected device:", device)

# Test small tensor on device
try:
    x = torch.randn(2, 2).to(device)
    print("Tensor device:", x.device)
    print("Device test: SUCCESS")
except Exception as e:
    print("Device test failed:", e)