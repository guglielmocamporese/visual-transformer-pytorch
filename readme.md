# PyTorch Implementation of the Visual Transformer (ViT) from Scratch
 Reimplementation of the paper: 
 
 "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale", *Dosovitskiy et al*, 2020. 
 
 [![arXiv](https://img.shields.io/badge/arXiv-2010.1192-red)](https://arxiv.org/abs/2010.1192)
 
 ![alt text](vit.gif "model_diagram")
 
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

# Train

```sh
$ python main.py \
    --mode "train" \
    --model "vit-base" \
    --patch_size 8 \
    --lr 3e-4 \
    --epochs 100
```

# Test

```sh
$ python main.py \
    --mode "test" \
    --model "vit-base" \
    --patch_size 8 \
    --model_checkpoint "./checkpoints/vit_base.ckpt"
```
