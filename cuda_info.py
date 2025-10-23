import torch

print("[INFO] CUDA Available:", torch.cuda.is_available())
print("[INFO] GPU Mame:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("[INFO] PyTorch CUDA version:", torch.version.cuda)