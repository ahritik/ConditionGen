import torch
print("MPS available:", torch.backends.mps.is_available())
print("CUDA available:", torch.cuda.is_available())
