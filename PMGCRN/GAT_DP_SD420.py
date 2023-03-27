"""Torch modules for graph attention networks(GAT)."""
from random import random

import torch
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn

import dgl.function as fn
from dgl.nn.pytorch.softmax import edge_softmax
from dgl.utils import expand_as_pair
from sentiment import compute_sentiment


# pylint: disable=W0235
class Identity(nn.Module):
    """A placeholder identity operator that is argument-insensitive.
    (Identity has already been supported by PyTorch 1.2, we will directly
    import torch.nn.Identity in the future)
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        """Return input"""
        return x


# pylint: enable=W0235
class GATConv(nn.Module):
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
    in_feats : int, or pair of ints
        Input feature size.

        If the layer is to be applied to a unidirectional bipartite graph, ``in_feats``
        specifies the input feature size on both the source and destination nodes.  If
        a scalar is given, the source and destination node feature size would take the
        same value.
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
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats

        self.fc = nn.Linear(self._in_src_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.relu = nn.ReLU()
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:  # bipartite graph neural networks
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in_{src}})` and :math:`(N_{out}, D_{in_{dst}})`.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()

        # print(11)
        h_src = h_dst = self.feat_drop(feat)
        feat_src = feat_dst = self.fc(h_src).view(
            -1, self._num_heads, self._out_feats)
        # print("h_src.shape{}".format(h_src.shape))
        # NOTE: GAT paper uses "first concatenation then linear projection"
        # to compute attention scores, while ours is "first projection then
        # addition", the two approaches are mathematically equivalent:
        # We decompose the weight vector a mentioned in the paper into
        # [a_l || a_r], then
        # a^T [Wh_i || Wh_j] = a_l Wh_i + a_r Wh_j
        # Our implementation is much efficient because we do not need to
        # save [Wh_i || Wh_j] on edges, which is not memory-efficient. Plus,
        # addition could be optimized with DGL's built-in function u_add_v,
        # which further speeds up computation and saves memory footprint.

        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)

        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)

        #print(er.shape, er.dtype)
        # print(feat_src.shape)
        # 版本1
        graph.srcdata.update({'ft': feat_src, 'el': el})
        graph.dstdata.update({'ftd': feat_dst, 'er': er})
        # dp计算

        graph.apply_edges(fn.u_dot_v('ft', 'ftd', 'edp'))

        # def cosine_function(edges):
        #     cosine = nn.functional.cosine_similarity(edges.src['ft'], edges.dst['ftd'], dim=2, eps=1e-8)
        #     e = edges.src['el']+edges.dst['er']
        #     return {'edp_ftfd': cosine,'e':e}
        # graph.apply_edges(cosine_function)
        # 取模
        efst = 1e-3
        ft = graph.srcdata['ft'].squeeze()
        ft = ft * ft
        ft = ft.sum(dim=1)
        ft[ft < efst] = efst
        # print(ft)
        ftd = graph.srcdata['ftd'].squeeze()
        ftd = ftd * ftd
        ftd = ftd.sum(dim=1)
        ftd[ftd < efst] = efst

        graph.srcdata.update({'ftt': 1 / (ft.unsqueeze(dim=1).unsqueeze(dim=2))})
        graph.dstdata.update({'ftdd': 1 / (ftd.unsqueeze(dim=1).unsqueeze(dim=2))})
        graph.apply_edges(fn.u_dot_v('ftt', 'ftdd', 'edp_ftfd'))

        # print(graph.edata['edp_ftfd'])
        # #原有计算
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        edp = self.relu(graph.edata['edp'] * graph.edata['edp_ftfd']).squeeze()
        # print(graph.edata['edp_ftfd'])
        # print(edp)
        # print(edp.shape)
        graph.edata.pop('edp_ftfd')
        graph.edata.pop('edp')
        e = graph.edata['e'].squeeze()
        graph.edata.pop('e')

        graph.edata['edp1'] = th.mul(edp, e)

        e = (graph.edata.pop('edp1'))
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.dstdata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h_dst).view(h_dst.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
