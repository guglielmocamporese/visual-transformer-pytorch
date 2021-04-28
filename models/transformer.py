##################################################
# Imports
##################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import math
from torch.optim import Adam


##################################################
# Utils
##################################################

def attention(q, k, v, mask=None):
    """
    Implementation of the cross-attention and the self-attention.
    Args:
        q: tensor of shape [B, T_Q, D_K]
        k: tensor of shape [B, T_V, D_K]
        v: tensor of shape [B, T_V, D_V]

    Output:
        out: tensor of shape [B, T_Q, D_v]
    """
    B = q.shape[0]
    scale = math.sqrt(k.shape[2])
    att = torch.bmm(q, k.transpose(1, 2)) / scale # [B, T_Q, T_V]
    if mask is not None:
        mask = mask.unsqueeze(0).repeat(B, 1, 1)
        att = torch.where(mask > 0.0, att, - math.inf * torch.ones_like(att))
    att = F.softmax(att, 2)
    out = torch.bmm(att, v)
    return out

def create_causal_mask(size1, size2):
    mask = torch.ones(size1, size2)
    mask = torch.triu(mask, diagonal=0)
    return mask


##################################################
# Head
##################################################

class Head(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762

    ::
             Linear proj.     Linear proj.     Linear proj.
               (query: q)       (key: k)        (value: v)
                  ↓                ↓                ↓
                   --------        |        --------
                           ↓       ↓       ↓
                          Attention (q, k, v)

    """
    def __init__(self, h_dim):
        super(Head, self).__init__()
        self.q_lin = nn.Linear(h_dim, h_dim, bias=False)
        self.k_lin = nn.Linear(h_dim, h_dim, bias=False)
        self.v_lin = nn.Linear(h_dim, h_dim, bias=False)

    def forward(self, q, k=None, v=None, mask=None):
        if k is None:
            k = q
        if v is None:
            v = k
        q = self.q_lin(q)
        k = self.k_lin(k)
        v = self.v_lin(v)
        x = attention(q, k, v, mask=mask)
        return x


##################################################
# Multi Head Attention
##################################################

class MultiHeadAttention(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    ::

            [Head_1, Head_2, ..., Head_h]
                           ↓
                       Cat (dim=2)
                           ↓
            Linear (in=h * h_dim, out=h_dim)

    """
    def __init__(self, h_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.h_dim = h_dim
        self.num_heads = num_heads
        self.heads = nn.ModuleList([
            Head(h_dim) for _ in range(num_heads)
        ])
        self.linear = nn.Linear(h_dim * num_heads, h_dim)

    def forward(self, q, k=None, v=None, mask=None):
        x = [head(q, k, v, mask=mask) for head in self.heads]
        x = torch.cat(x, -1) # [B, T, h_dim * num_heads]
        x = self.linear(x) # [B, T, h_dim]
        return x


##################################################
# Transformer Encoder Layer
##################################################

class TransformerEncoderLayer(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, h_dim, num_heads, d_ff=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm1 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, h_dim)
        )
        self.norm2 = nn.LayerNorm(h_dim)

    def forward(self, x, mask=None):
        x = self.mha(x, x, x, mask=mask) + x
        x = self.norm1(x)
        x = self.ffn(x) + x
        x = self.norm2(x)
        return x


##################################################
# Transformer Encoder
##################################################

class TransformerEncoder(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048, 
                 max_time_steps=None, use_clf_token=False):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(h_dim, num_heads, d_ff=2048) 
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
# Transformer Decoder Layer
##################################################

class TransformerDecoderLayer(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, h_dim, num_heads, d_ff=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.mask_mha = MultiHeadAttention(h_dim, num_heads)
        self.norm1 = nn.LayerNorm(h_dim)
        self.mha = MultiHeadAttention(h_dim, num_heads)
        self.norm2 = nn.LayerNorm(h_dim)
        self.ffn = nn.Sequential(
            nn.Linear(h_dim, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, h_dim)
        )
        self.norm3 = nn.LayerNorm(h_dim)

    def forward(self, x, x_enc):
        mask = create_causal_mask(x.shape[1], x.shape[1]).to(x.device)
        x = self.mask_mha(x, x, x, mask=mask) + x
        x = self.norm1(x)
        x = self.mha(x, x_enc, x_enc, mask=mask) + x
        x = self.norm2(x)
        x = self.ffn(x) + x
        x = self.norm3(x)
        return x


##################################################
# Transformer Decoder
##################################################

class TransformerDecoder(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, num_layers, h_dim, num_heads, d_ff=2048, 
                 max_time_steps=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(h_dim, num_heads, d_ff=2048) 
            for _ in range(num_layers)
        ])
        self.pos_emb = nn.Embedding(max_time_steps, h_dim)

    def forward(self, x, x_enc):
        embs = self.pos_emb.weight[:x.shape[1]]
        x += embs
        for layer in self.layers:
            x = layer(x, x_enc)
        return x


##################################################
# Transformer
##################################################

class Transformer(nn.Module):
    """
    Attention is all you need, Vaswani et al, 2017.
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, num_layers_enc, num_layers_dec, h_dim, num_heads, 
                 num_classes, d_ff=2048, max_time_steps_enc=None, 
                 max_time_steps_dec=None):
        super(Transformer, self).__init__()
        self.enc = TransformerEncoder(num_layers_enc, h_dim, num_heads, d_ff, 
                                      max_time_steps_enc)
        self.dec = TransformerDecoder(num_layers_dec, h_dim, num_heads, d_ff, 
                                      max_time_steps_dec)
        self.linear = nn.Linear(h_dim, num_classes)

    def forward(self, x_enc, x_dec, mask_enc=None):
        x_enc = self.enc(x_enc, mask=mask_enc)
        x_dec = self.dec(x_dec, x_enc)
        x = self.linear(x_dec)
        return x

    def generate(self, x_enc, steps, mask_enc=None):
        """
        x_enc: tensor of shape [1, T, h_dim]
        steps has to be less than max_time_steps_dec.
        """
        self.eval()
        out = torch.zeros(steps, self.dec.h_dim, device=x_enc.device)
        for i in range(steps):
            logits_dec = self(x_enc, out)
            preds_dec = F.softmax(logits_dec, -1)
            #out[:, i] = torch.multinomial(preds_dec[:, i], 1)
            # TODO...


