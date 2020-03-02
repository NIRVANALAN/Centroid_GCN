
"""Torch modules for graph attention networks(GAT)."""
# pylint: disable= no-member, arguments-differ, invalid-name
import time
import torch as th
from torch import nn
import torch.nn.functional as F

import dgl.function as fn
from dgl.nn.pytorch import edge_softmax
from dgl.nn.pytorch.utils import Identity

from .utils import timeit

# pylint: enable=W0235


class CentroidGATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(CentroidGATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(
            th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(
                    in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.centroid_activation = F.tanh

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat, cluster_id=None, cluster_centroid=None, stat=None):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()  # return as graph can be used in fcuntion scope
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        # if cluster_id is not None:
        #     # start = time.time()
        #     cluster_centroid = cluster_centroid.view(
        #         -1, self._num_heads, self._out_feats)  # [6*8*8] * [1,8,8]
        #     # el = cluster_centroid * self.attn_l[cluster_id].sum(-1).unsqueeze(-1)
        #     el = (cluster_centroid * self.attn_l)
        #     er = (cluster_centroid * self.attn_r)
        #     # er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        #     # print(f'cluster el/er calculation time: {time.time() - start:.7f}')
        #     er = er[cluster_id].sum(dim=-1).unsqueeze(-1)
        #     el = el[cluster_id].sum(dim=-1).unsqueeze(-1)
        #     # el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        # else:
        # start = time.time()

        el = (feat * self.attn_l)  # [3708*8*8] * [1,8,8]
        er = (feat * self.attn_r)
        # print(f'el/er calculation time: {time.time() - start:.7f}')
        el = el.sum(dim=-1).unsqueeze(-1)
        er = er.sum(dim=-1).unsqueeze(-1)

        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(
            edge_softmax(graph, e))  # scale after softmax
        if stat is not None:
            stat.append(graph.edata['a'].detach().cpu().numpy())
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        embedding = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            embedding = embedding + resval
        # activation
        if self.activation:
            return self.activation(embedding), self.activation(embedding)
            # return self.activation(embedding), embedding
            # return rst, embedding
        elif stat is not None:
            return embedding, stat
        else:
            return embedding  # output logits
