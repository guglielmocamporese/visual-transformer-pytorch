# Implementation of the Visual Transformer (ViT) from Scratch

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

## Model Configurations

From the paper [[link](https://arxiv.org/abs/2010.11929)]:

| Model     | Layers | Hidden Size | MLP Size | Heads | Params | 
| --------- | ------ | ----------- | -------- | ----- | ------ |
| ViT-Base  | 12     | 768         | 3072     | 12    | 86 M   |
| ViT-Large | 24     | 1024        | 4096     | 16    | 307 M  |
| ViT-Huge  | 32     | 1280        | 5120     | 16    | 632 M  |
