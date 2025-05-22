__all__ = ['PatchTST_backbone']

import argparse
# Cell
from typing import Callable, Optional
import torch
from mamba_ssm import Mamba

from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.Embed import DataEmbedding_inverted
from layers.Mamba_EncDec import EncoderLayer, Encoder, PatchEncoderLayer
from layers.RevIN import RevIN
from utils.lead_estimate import compress_fft_features, shifted_leader_seq
import json
# from collections import OrderedDict


'''
B,T,N to B,N,P,L into mamba then return B,N,target_len
B,N,K,target_len ---> into mamba / or use FFT or others then into mamba
want to add gate between Plugin & Patch
'''
class Model(nn.Module):
    def __init__(self, config):

        super(Model, self).__init__()
        self.model = Pamba_backbone(config)

    def forward(self, x, x_mark_enc, x_dec, x_mark_dec):  # x: [Batch, Input length, Channel]
        x = self.model(x, x_mark_enc, x_dec, x_mark_dec)
        return x


class Pamba_backbone(nn.Module):
    def __init__(self, config):

        super().__init__()
        self.revin = config.revin
        self.revin_layer = RevIN(config.enc_in, affine=True, subtract_last=False)

        self.pred_len = config.pred_len

        # Patching
        self.patch_len = config.patch_len
        self.stride = config.stride

        self.stride = config.stride
        self.padding_patch = config.padding_patch
        patch_num = int((config.seq_len - config.patch_len) / config.stride + 1)
        if config.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, config.stride))
            patch_num += 1

        # Backbone
        self.backbone = PatchEncoder(config)
        self.n_vars = config.enc_in
        self.head_number = config.d_model * (patch_num)
        self.head = Flatten_Head( self.head_number,  config.d_model,head_dropout=config.head_dropout)  # output is dim or other design need to try
        self.patch_ffn = nn.Linear(config.d_model, config.pred_len)

        # Token Embedding
        self.embed_size = config.embed_size  # embed_size
        self.token_embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.linear = nn.Linear(config.d_model, config.embed_size)
        self.softmax = nn.Softmax(dim=-1)
        self.linear2 = nn.Linear(config.d_model*(self.embed_size+1), config.d_model*(self.embed_size+1))

        #value header
        self.gate = nn.Linear((self.embed_size+1) * config.d_model,(self.embed_size+1) * config.d_model)
        self.n_vars_header = nn.Linear((self.embed_size+1) * config.d_model, self.pred_len)
        self.act = nn.SiLU()

        self.dropout = nn.Dropout(config.head_dropout)

        #HORIZOM
        self.view_1 = nn.Linear(config.seq_len, config.d_model)#B,T,1 --->B,1,D
        self.view_2 = nn.Linear(config.enc_in, 1) #B,T,N --->B,T,1
        self.view_3 = nn.Linear(config.seq_len, config.d_model)



        self.d_model = config.d_model * (self.embed_size +1)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        # d_model=config.pred_len*(),  # Model dimension d_model
                        d_model=self.d_model,
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        # d_model=config.pred_len*(),  # Model dimension d_model
                        d_model=self.d_model,
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    self.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

    '''
    input B,C,T output B,C,H,T
    '''

    def tokenEmb(self, x): # x: [Batch, Channel , input len]

        u = x.unsqueeze(3) # B,N,D---->B,N,D,1
        y = self.token_embeddings
        out = u * y #B,N,D,1 ----> B,N,D,E
        out = out.permute(0, 1, 3, 2) # B,N,E,D
        B, N, E, T = out.shape

        #classify get the max value
        b = self.linear(x)  # B,N,E
        b = self.softmax(b)  # x: [B,N,embed_size]
        b_max = torch.argmax(b, dim=2, keepdim=True)  # x: [B,N,1]
        b_max = b_max.squeeze(-1)   # x: [B,N]
        b_max = b_max + 1

        # Mask the data
        e_indices = torch.arange(E, device=out.device).view(1, 1, E)  # 形状 (1, 1, E)
        e_indices_expanded = e_indices.expand(B, N, E)  # 扩展到 (B, N, E)

        threshold = torch.where(b_max == E, E, b_max).unsqueeze(-1)  # 形状 (B, N, 1)

        mask = (e_indices_expanded < threshold).unsqueeze(-1)  # 扩展维度到 (B, N, E, 1)

        masked_out = torch.where(mask, out, torch.zeros_like(out))

        return masked_out  # output x: [Batch, Channel ,H, input len]

    def horizon(self, x):  # x B,T,N
        p = self.view_2(x).permute(0, 2, 1)  # B,T,N --->B,T,1--->B,1,T
        p = self.view_1(p)  # B,1,T----> B,1,D

        n = x.permute(0, 2, 1)
        n = self.view_3(n)  # x B,N,D
        n = n.unsqueeze(2)  # x B,N,1,D

        return p, n  # p:B,1,D n B,N,1,D

    def forward(self, z, x_mark_enc, x_dec, x_mark_dec):  # input [bs x  seq_len x n_vars]
        # norm
        B, T, N = z.shape
        if self.revin:
            # Normalization from Non-stationary Transformer
            means = z.mean(1, keepdim=True).detach()
            z = z - means
            stdev = torch.sqrt(torch.var(z, dim=1, keepdim=True, unbiased=False) + 1e-5)
            z /= stdev

        pview,nview = self.horizon(z)  # B,T,N ---- B,1,D

        z  = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]

        z = self.backbone(z,pview)  # input z: [bs x nvars  x patch_num x d_model] output z: [bs x nvars x d_model x patch_num]

        z = self.head(z)  # z: [bs x nvars x d_model]

        out = self.tokenEmb(z)  # out: [bs x nvars x Hidden x target_window]
        out = torch.cat((nview, out), dim=2)  # out: [bs x nvars x (Hidden+K+1) x target_window]

        out = torch.reshape(out, (out.shape[0], out.shape[1], out.shape[2] * out.shape[3]))  # out: [bs x nvars x (Hidden+K+1) * d_model]
        out = self.act(self.linear2(out))
        out, _ = self.encoder(out)

        out = self.act(self.gate(out)) * out
        out = self.n_vars_header(out)
        out = self.dropout(out)
        z = self.patch_ffn(z) + out

        z = z.permute(0, 2, 1)  # z: [bs x target_window x nvars]

        # denorm
        if self.revin:
            z = z * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            z = z + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return z


class Flatten_Head(nn.Module):
    def __init__(self, nf, d_model, head_dropout=0.2):
        super().__init__()

        self.gate = nn.Linear(nf, nf)
        self.act = nn.SiLU()

        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, d_model)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)  # bs x nvars x d_model * patch_num
        z = self.act(self.gate(x))
        x = x * z
        x = self.linear(x)
        x = self.dropout(x)

        return x


class PatchEncoder(nn.Module):  # i means channel-independent
    def __init__(self, config):
        super().__init__()

        self.W_P = nn.Linear(config.patch_len,
                             config.d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Encoder
        self.encoder = self.encoder = Encoder(
            [
                PatchEncoderLayer(
                    Mamba(
                        d_model=config.d_model,  # Model dimension d_model
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=config.d_model,  # Model dimension d_model
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    config.d_model,
                    config.d_ff,
                    dropout=config.dropout,
                    activation=config.activation
                ) for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)

        )

    def forward(self, x,view) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        b, n, p, d = x.shape

        u = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))  # u: [bs x nvars * patch_num x d_model]
        u = self.dropout(u)  # u: [bs x nvars * patch_num x d_model]
        u = torch.cat((u,view),dim=1) # u: [bs x nvars * patch_num +1 x d_model]

        # Encoder
        z,_ = self.encoder(u)  # z: [bs x nvars * patch_num +1 x d_model]
        z = z[:, :-1, :]  # z: [bs x nvars * patch_num x d_model]

        z = torch.reshape(z, (b, n, p, d))  # z: [bs x nvars x patch_num x d_model]

        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        return z


