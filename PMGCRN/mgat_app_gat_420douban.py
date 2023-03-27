import torch
from torch import nn
from torch.nn import init
import torch.nn.functional as F
from dgl import function as fn
from dgl.base import DGLError
from GAT_DP_SD420 import GATConv
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Dropout(0.6),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        self.attn_drop = nn.Dropout(0.6)

    def forward(self, z):
        w = self.project(z)
        # print(w.shape)
        w = w.mean(0)  # (M, 1)
        # print(w.shape)
        # beta = self.attn_drop(torch.softmax(w, dim=0))                 # (M, 1)
        beta = torch.softmax(w, dim=0)
        beta1 = beta

        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)

        return (beta * z).sum(1), beta1  # (N, D * K)


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        # nn.init.constant_(m.bias, 0)


class MMGATLayer(nn.Module):

    def __init__(self, in_feats, out_feats,
                 activation=None, dropout=0.3):
        super(MMGATLayer, self).__init__()
        self.norm = nn.BatchNorm1d(in_feats)
        self.num_layers = 3
        self.gat_layers = nn.ModuleList()
        self.activation = activation

        self.residual = False
        self.gat_layers.append(GATConv(
            in_feats, out_feats, 1,
            feat_drop=dropout, attn_drop=dropout, negative_slope=0.2, residual=self.residual,
            activation=self.activation))
        # hidden layers
        for l in range(1, self.num_layers - 1):
            # due to multi-head, the in_dim = num_hidden * num_heads
            self.gat_layers.append(GATConv(
                out_feats, out_feats, 1,
                feat_drop=dropout, attn_drop=dropout, negative_slope=0.2, residual=self.residual,
                activation=self.activation))
        # output projection
        # self.gat_layers.append(GATConv(
        #     num_hidden * heads[-2], num_classes, heads[-1],
        #     feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(
            out_feats, out_feats, 1,
            feat_drop=dropout, attn_drop=dropout, negative_slope=0.2, residual=self.residual,
            activation=None))
        # output projection

        self.attention = SemanticAttention(64, 16)
        self.attention.apply(weight_init)
        self.batchNorm = nn.BatchNorm1d(64)

    def forward(self, g):
        self.g = g
        h = g.srcdata['h']

        h_ = None
        a_ = None
        # h = F.normalize(h)

        h_list = []
        # h = F.normalize(h)
        # h_ =  h
        # for l in range(self.num_layers):
        #     h = F.normalize(h)
        #     h = self.gat_layers[l](self.g, h).flatten(1)
        #     # a_.append( self.g.edata['a'])
        #     # if l == 0:
        #     #     h_ = h
        #     #     # a_ = self.g.edata['a']
        #     # else:
        #     h_ = h_ + h
        #     # h_[:6038] = h_[:6038] + h[:6038]
        #     h_[6038:] = h_[6038:] - h[6038:]
        #
        #         # print("h_shape{}".format(h_.shape))
        #         # print("hshape{}".format(h.shape))
        #     h = h_
        for l in range(self.num_layers):
            h = F.normalize(h)
            h = self.gat_layers[l](self.g, h).flatten(1)
            # a_.append( self.g.edata['a'])
            if l == 0:
                h_ = h
                # a_ = self.g.edata['a']
            else:
                h_ = h_ + h
                # h_[:6038] = h_[:6038] - h[:6038]
                # h_[6038:] = h_[6038:] - h[6038:]

                # print("h_shape{}".format(h_.shape))
                # print("hshape{}".format(h.shape))
            h = h_
            # h = F.normalize(h)
            h_list.append(h)
            # a_ = a_ + self.g.edata['a']
        h_array = torch.stack(h_list, dim=1)
        # print(h_array.shape)
        h_, attention_value = self.attention(h_array)
        # print(h_)
        # attention_value = 1
        h_ = F.normalize(h_)
        return h_, attention_value
