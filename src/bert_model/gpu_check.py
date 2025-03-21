import torch
print(torch.cuda.is_available())  # Should return True if GPU is available
print(torch.cuda.current_device())  # Should return the current device index (0 for the first GPU)
print(torch.cuda.get_device_name(0))  # Should return the GPU name (e.g., "NVIDIA GeForce")
