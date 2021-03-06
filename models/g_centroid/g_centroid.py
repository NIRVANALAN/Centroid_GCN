"""
Graph Attention Networks in DGL using SPMV optimization.
References
----------
Paper: https://arxiv.org/abs/1710.10903
Author's code: https://github.com/PetarV-/GAT
Pytorch implementation: https://github.com/Diego999/pyGAT
"""

import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from .CentroidConv import CentroidGATConv


class GATCentroid(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation,
                 feat_drop,
                 attn_drop,
                 negative_slope,
                 residual, batch_norm=False):
        super(GATCentroid, self).__init__()
        self.g = g
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(CentroidGATConv(
            in_dim, num_hidden, heads[0],
            feat_drop, attn_drop, negative_slope, False, self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(CentroidGATConv(
                num_hidden * heads[l-1], num_hidden, heads[l],
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output projection
        self.gat_layers.append(CentroidGATConv(
            num_hidden * heads[-2], num_classes, heads[-1],
            feat_drop, attn_drop, negative_slope, residual, None))
        if batch_norm:
            self.batch_norm = [
                nn.BatchNorm1d(num_hidden, affine=False, track_running_stats=False)] * num_layers
        else:
            self.batch_norm = None

    def forward(self, feat, cluster_id=None, cluster_centroid=None, stat=None):
        h = feat
        for l in range(self.num_layers):
            h, embedding = self.gat_layers[l](self.g, h, cluster_id,
                                              cluster_centroid, stat)
            h = h.flatten(1)
            if getattr(self, 'batch_norm'):
                h = self.batch_norm[l](h)
            embedding = embedding.flatten(1)
        # output projection
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits, embedding  # n * 64
