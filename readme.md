# Implementation of the Visual Transformer (ViT) from scratch

ongoing...

## Usage

```python
# Imports
import torch
from models.vit import ViT

# Create the model
vit = ViT(
    patch_size=4, 
    num_layers=2, 
    h_dim=256, 
    num_heads=8, 
    num_classes=10, 
    d_ff=2048, 
    max_time_steps=1000, 
    use_clf_token=True,
)

# Inference
model.eval()
x = torch.randn(1, 3, 32, 32) # [B, C, H, W]
logits = vit(x) # [B, N_CL]
```
