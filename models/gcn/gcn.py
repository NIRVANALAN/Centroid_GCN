"""GCN using DGL nn package

References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_dim,
                 num_hidden,
                 num_classes,
                 num_layers,
                 activation,
                 dropout=0.5, **kwargs):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConv(in_dim, num_hidden, activation=activation))
        # hidden layers
        for i in range(num_layers - 1):
            self.layers.append(GraphConv(num_hidden, num_hidden, activation=activation))
        # output layer
        self.layers.append(GraphConv(num_hidden, num_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
        return h
