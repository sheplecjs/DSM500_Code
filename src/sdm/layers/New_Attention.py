import torch
from torch import nn
import numpy as np
from scipy.special import softmax
import random

import torch.nn.functional as F

import matplotlib.pyplot as plt

import numpy as np
import math
from math import sqrt
from sdm.utils.masking import TriangularCausalMask, ProbMask
from .SelfAttention_Family import FullAttention
import os

class RNDAttention(nn.Module):
    """A random attention layer.
    """
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(RNDAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        # _, S, _, D = values.shape
        # scale = self.scale or 1. / sqrt(E)

        # print(queries.shape, values.shape)
        # torch.Size([32, 48, 8, 64]) torch.Size([32, 48, 8, 64])

        # scores = torch.rand([B, H, L, L], dtype=torch.float)
        # score shape = [32, 8, 48, 48]

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

        #     scores.masked_fill_(attn_mask.mask, -np.inf)

        # A = self.dropout(torch.softmax(scale * scores, dim=-1))
        # V = torch.einsum("bhls,bshd->blhd", A, values)

        # print(V, V.shape)
        V = torch.rand([B, L, H, E], dtype=torch.float)
        # should be torch.Size([32, 48, 8, 64])

        # one = torch.sum(V[:, 0, 0, 0])
        # two = torch.sum(V[0, :, 0, 0])
        # three = torch.sum(V[0, 0, :, 0])
        # four = torch.sum(V[0, 0, 0 :])

        # with open("attn.txt", "w") as text_file:
        #     text_file.write(f"Attn tensor: {V}")

        if self.output_attention:
            return (V.contiguous(), V)
        else:
            return (V.contiguous(), None)


class SDMAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False, max_lags=10):
        super(SDMAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.max_lags = max_lags

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape

        # batch weights
        W = []
        softm = torch.nn.Softmax(dim=-1)

        for batch in values:
            
            # split series for comparison
            x = batch[:24]
            y = batch[24:]
            
            v = torch.rand(x.shape)
            u = torch.rand(x.shape)

            
            # create a randomly initialized weights tensor    
            w = softm(torch.rand((x.shape[0], x.shape[0], x.shape[2])))
            
            # initialized distance tensor
            d = torch.rand((x.shape[0], x.shape[1], x.shape[2]))
            
            # outer iteration
            for s in range(len(x)):
                
                omega = torch.median(v)
                lamb = torch.mean(u) ** 2
                    
                g = 0
                
                for i in reversed(range(len(x) -1)):
                    w[i][s] = w[i][s - 1] + (omega / 3 * (w[(i - 1)][s - 1] + w[i][s - 1] + w[(i + 1)][s]))

                for _v in range(len(x)):
                    d[_v] = y[_v] - x[_v]
                    w_hat = lamb * (math.exp(-lamb * d[_v][i][s]))
                    g = g + w_hat
                
                v[s] = 0
                u[s] = 0
                
                
                for i in range(len(x)):
                    
                    w[i][s] = w_hat / g
                    v[s] = v[s] + abs(w[i][s] - w[i][s - 1])
                    u[s] = u[s] + (w[i][s - 1] * d[i])
                    
            vanishing = torch.full((24, 24, 512), 1e-9)
            final = torch.cat((vanishing, w), 0)
            m = torch.mean(final, dim=1)
            W.append(final)
            
        scores = torch.stack(W)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        if self.output_attention:
            return (scores, scores)
        else:
            return (scores, None)