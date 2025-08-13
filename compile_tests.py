import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoModel

import time
import os


"""
# -------------------------------
# 1. Load model and compile
# -------------------------------
model = AutoModel.from_pretrained(
    "google/vit-base-patch16-224", 
    local_files_only=True
).cuda()


starttime = time.time()
model = torch.compile(
    model,
    mode="reduce-overhead",  # minimal fusion, fast compile
    backend="aot_eager"
)

print(f"Compile time: {time.time() - starttime} seconds")

# -------------------------------
# 2. Warmup with a small dummy batch
# -------------------------------
starttime = time.time()
dummy = torch.randn(2, 3, 224, 224, device="cuda")
_ = model(dummy)  # triggers kernel compilation / fusion

print(f"Dummy batch time: {time.time() - starttime} seconds")


model_path = os.path.join("/scratch", "mcatchen", "precompiled_vit.pt")
torch.save(model, model_path)
"""
# load that bad boy

model_path = os.path.join("/scratch", "mcatchen", "precompiled_vit.pt")
model = torch.load(model_path, weights_only=False)

starttime = time.time()
dummy = torch.randn(2, 3, 224, 224, device="cuda")
_ = model(dummy)  # triggers kernel compilation / fusion
print(f"Dummy batch time: {time.time() - starttime} seconds")

starttime = time.time()
for i in range(64):
    starttime = time.time()
    dummy = torch.randn(2, 3, 224, 224, device="cuda")
    _ = model(dummy)  # triggers kernel compilation / fusion

print(f"Multi dummy batch time: {time.time() - starttime} seconds")
