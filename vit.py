##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import Adam

from transformer import MultiHeadAttention

# Utils
class Transpose(nn.Module):
    def __init__(self, d0, d1): 
        super(Transpose, self).__init__()
        self.d0, self.d1 = d0, d1

    def forward(self, x):
        return x.transpose(self.d0, self.d1)


##################################################
# ViT Transformer Encoder Layer
##################################################

class ViTransformerEncoderLayer(nn.Module):
    def __init__(self, h_dim, num_heads, d_ff=2048):
        super(ViTransformerEncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, h_dim)
        )

    def forward(self, x, mask=None):
        x_ = self.norm1(x)
        x = self.mha(x_, x_, x_, mask=mask) + x
        x_ = self.norm2(x)
        x = self.ffn(x_) + x
        return x


##################################################
# Vit Transformer Encoder
##################################################

class ViTransformerEncoder(nn.Module):
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048, 
                 max_time_steps=None, use_clf_token=False):
        super(ViTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTransformerEncoderLayer(h_dim, num_heads, d_ff=2048) 
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Embedding(max_time_steps, h_dim)
        self.use_clf_token = use_clf_token
        if self.use_clf_token:
            self.clf_token = nn.Parameter(torch.randn(1, h_dim))

    def forward(self, x, mask=None):
        if self.use_clf_token:
            clf_token = self.clf_token.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = torch.cat([clf_token, x], 1)
            if mask is not None:
                raise Exception('Error. clf_token with mask is not supported.')
        embs = self.pos_emb.weight[:x.shape[1]]
        x += embs
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


##################################################
# Visual Transformer (ViT)
##################################################

class ViT(nn.Module):
    def __init__(self, patch_size, num_layers, h_dim, num_heads, num_classes, 
                 d_ff=2048, max_time_steps=None, use_clf_token=True):
        super(ViT, self).__init__()
        self.proc = nn.Sequential(
            nn.Unfold((patch_size, patch_size), 
                      stride=(patch_size, patch_size)),
            Transpose(1, 2),
            nn.Linear(3 * patch_size * patch_size, h_dim),
        )
        self.enc = ViTransformerEncoder(num_layers, h_dim, num_heads, 
                                         d_ff=d_ff, 
                                         max_time_steps=max_time_steps, 
                                         use_clf_token=use_clf_token)
        self.mlp = nn.Linear(h_dim, num_classes)

    def forward(self, x):
        x = self.proc(x)
        x = self.enc(x)
        x = x[:, 0] if self.enc.use_clf_token else x.mean(1)
        x = self.mlp(x)
        return x


# Get visual transformer
def get_vit(args):
    model_args = {
        'patch_size': args.patch_size, 
        'num_layers': args.num_layers, 
        'h_dim': args.h_dim, 
        'num_heads': args.num_heads, 
        'num_classes': args.num_classes, 
        'd_ff': args.d_ff, 
        'max_time_steps': args.max_time_steps, 
        'use_clf_token': args.use_clf_token,
    }
    model = ViT(**model_args)
    if len(args.model_checkpoint) > 0:
        model = model.load_from_checkpoint(args.model_checkpoint, **model_args)
        print(f'Model checkpoint loaded from {args.model_checkpoint}')
    return model


##################################################
# Classifier - PyTorchLightning Wrapper for ViT
##################################################

class Classifier(pl.LightningModule):
    def __init__(self, args):
        super(Classifier, self).__init__()
        self.args = args
        self.model = get_vit(args)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx, part='train'):
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        acc = (1.0 * (F.softmax(logits, 1).argmax(1) == y)).mean()

        self.log(f'{part}_loss', loss, prog_bar=True)
        self.log(f'{part}_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx, part='val')

    def configure_optimizers(self):
        return Adam(self.parameters(), self.args.lr)
