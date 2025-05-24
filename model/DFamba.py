
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
        self.backbone = TSTiEncoder(config)

        self.individual = config.individual
        self.n_vars = config.enc_in
        self.head_nf = config.d_model * (patch_num)

        self.head = Flatten_Head(self.individual, self.n_vars, self.head_nf, config.pred_len, config.d_model,
                                 config.enc_in,
                                 head_dropout=config.head_dropout)  # output is dim or other design need to try
        self.patch_header = nn.Linear(config.pred_len, self.pred_len)

        self.K = config.K

        self.embed_size = config.embed_size  # embed_size
        self.token_embeddings = nn.Parameter(torch.randn(1, self.embed_size))
        self.channel_embeddings = nn.Parameter(torch.randn(1, config.seq_len))

        self.gate = nn.Linear((self.embed_size + config.K  +1) * (config.d_model),
                              (self.embed_size + config.K  +1) * (config.d_model))
        self.n_vars_header = nn.Linear(config.d_model, self.pred_len)
        self.act = nn.SiLU()

        self.dropout = nn.Dropout(config.head_dropout)

        #HORIZOM

        self.view_1 = nn.Linear(config.enc_in, 1)
        self.view_2 = nn.Linear(config.seq_len, config.d_model)
        self.view_3 = nn.Linear(config.enc_in, 1)
        self.view_4 = nn.Linear(config.seq_len, config.d_model)

        self.patch_ffn = nn.Linear(config.d_model, config.pred_len)


        # self.d_model = config.d_model * (config.K + self.embed_size +1)
        self.d_model = config.d_model

        self.encoder = Encoder(
            [
                EncoderLayer(
                    Mamba(
                        # d_model=config.pred_len*(config.K),  # Model dimension d_model
                        d_model=self.d_model,
                        d_state=config.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        # d_model=config.pred_len*(config.K),  # Model dimension d_model
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

    # def tokenEmb(self, x):
    #     # x: [Batch, Channel , input len]
    #     x = x.unsqueeze(3)
    #     org = x.permute(0, 1, 3, 2)
    #     # N*T*1 x 1*D = N*T*D
    #     y = self.token_embeddings
    #     out = x * y
    #     out = out.permute(0, 1, 3, 2)
    #     out = torch.cat((out, org), dim=2)
    #     return out  # output x: [Batch, Channel ,H+1, input len]
    def tokenEmb(self, x):
        # x: [Batch, Channel , input len]
        x = x.unsqueeze(3)
        # N*T*1 x 1*D = N*T*D
        y = self.token_embeddings
        out = x * y
        out = out.permute(0, 1, 3, 2)
        return out  # output x: [Batch, Channel ,H, input len]

    def channelEmb(self, x):
        # x: [Batch, Channel , Seq]
        y = self.channel_embeddings
        out = x * y #x: [Batch, Channel ,Seq]
        return out+x  # output x: [Batch, Channel ,H, input len]

    def horizon(self,x): # x B,T,N
        p = self.view_1(x).permute(0,2,1)  # x B,1,T
        p_1 = self.view_2(p)  # x B,1,D

        n = self.view_3(x).permute(0,2,1)  # x B,1,T
        n_1 = self.view_4(n)  # x B,1,D

        return p_1,n_1 # p:B,1,D n B,N,1,D

    def forward(self, z, x_mark_enc, x_dec, x_mark_dec):  # input [bs x  seq_len x n_vars]
        # norm
        B, T, N = z.shape
        if self.revin:
            # Normalization from Non-stationary Transformer
            means = z.mean(1, keepdim=True).detach()
            z = z - means
            stdev = torch.sqrt(torch.var(z, dim=1, keepdim=True, unbiased=False) + 1e-5)
            z /= stdev

        pview, nview = self.horizon(z)  # B,T,N ---- B,1,D

        z = z.permute(0, 2, 1)

        # do patching
        if self.padding_patch == 'end':
            z = self.padding_patch_layer(z)

        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # z: [bs x nvars x patch_num x patch_len]

        z = self.backbone(z,pview)  # input z: [bs x nvars  x patch_num x d_model] output z: [bs x nvars x d_model x patch_num]

        z = self.head(z)  # z: [bs x nvars x d_model]

        out = torch.cat((z, nview), dim=1)  # out: [bs x nvars+1 x  target_window]

        out, _ = self.encoder(out)

        out = out[:, :-1, :]  # out: [bs x nvars x target_window]

        # r = self.patch_ffn(z) + self.n_vars_header(out)
        r = self.n_vars_header(z+out)

        r = r.permute(0, 2, 1)  # z: [bs x target_window x nvars]

        # denorm
        if self.revin:
            r = r * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            r = r + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return r


class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, d_model, enc_in, head_dropout=0):
        super().__init__()

        self.individual = individual
        self.n_vars = n_vars

        self.gate = nn.Linear(nf, nf)
        # self.act = nn.ReLU()
        # self.act = nn.SiLU()
        self.act = nn.SiLU()

        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, d_model)
            self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        # print('here comes head,x.shape=',x.shape)
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:, i, :, :])  # z: [bs x d_model * patch_num]
                z = self.linears[i](z)  # z: [bs x target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)  # x: [bs x nvars x target_window]
        else:
            x = self.flatten(x)  # bs x nvars x d_model * patch_num

            z = self.act(self.gate(x))
            x = x * z

            x = self.linear(x)
            x = self.dropout(x)


        return x


class TSTiEncoder(nn.Module):  # i means channel-independent
    def __init__(self, config):
        super().__init__()

        # Input encoding

        self.W_P = nn.Linear(config.patch_len,
                             config.d_model)  # Eq 1: projection of feature vectors onto a d-dim vector space

        # Residual dropout
        self.dropout = nn.Dropout(config.dropout)

        # Encoder
        self.encoder = TSTEncoder(config)

        self.patch_len = config.patch_len
        self.stride = config.stride

    def forward(self, x,view) -> Tensor:  # x: [bs x nvars x patch_len x patch_num]

        # n_vars = x.shape[1]


        # Input encoding
        # x = x.permute(0, 1, 3, 2)  # x: [bs x nvars x patch_num x patch_len]
        x = self.W_P(x)  # x: [bs x nvars x patch_num x d_model]
        b, n, p, d = x.shape

        # u = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))  # u: [bs x nvars * patch_num x d_model]
        # u = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))  # u: [bs x nvars * patch_num x d_model]
        u = torch.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], x.shape[3]))  # u: [bs x nvars * patch_num x d_model]
        u = self.dropout(u)  # u: [bs x nvars * patch_num x d_model]

        u = torch.cat((u,view),dim=1) # u: [bs x nvars * patch_num +1 x d_model]

        # # 随机打乱第二个维度?换个地方打乱比较好吧
        # indices = torch.randperm(u.size(1))  # 生成一个随机的 0 到 95 的排列
        #
        # # 按时间步的顺序打乱
        # u = u[:, indices, :]

        # Encoder
        z = self.encoder(u)  # z: [bs x nvars * patch_num +1 x d_model]

        z = z[:, :-1, :]  # z: [bs x nvars * patch_num x d_model]

        z = torch.reshape(z, (b, n, p, d))  # z: [bs x nvars x patch_num x d_model]
        # print('after mamba reshape,z.shape=', z.shape)
        z = z.permute(0, 1, 3, 2)  # z: [bs x nvars x d_model x patch_num]

        # Xig add on 20240910 for gate

        return z

    # Cell


class TSTEncoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        patch_num = int((configs.seq_len - configs.patch_len) / configs.stride + 1)
        if configs.padding_patch == 'end':  # can be modified to general case
            self.padding_patch_layer = nn.ReplicationPad1d((0, configs.stride))
            patch_num += 1

        self.encoder = Encoder(
            [
                PatchEncoderLayer(
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    Mamba(
                        d_model=configs.d_model,  # Model dimension d_model
                        d_state=configs.d_state,  # SSM state expansion factor
                        d_conv=2,  # Local convolution width
                        expand=1,  # Block expansion factor)
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)

        )

    def forward(self, x):
        x, _ = self.encoder(x)
        return x


