import torch.nn as nn
import torch

import mgat_app_gat_420douban
from functools import partial
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.nn import init
from lgcn420 import LNet


class Model(nn.Module):
    def __init__(self, num_nodes, in_dim, h_dim, out_dim, num_classes, num_rels,
                 num_hidden_layers=0, dropout=0.35, mmGATdropout=0.35, out_drop=0.65, residual=0.12, gat_type='sum',
                 data_file='result/', final_embed='IMDB_SUM', text_item=None, img_item=None):
        super(Model, self).__init__()
        # print('-----------这里是Model-----------')
        self.num_nodes = num_nodes
        # print('num_nodes:   ', self.num_nodes)
        self.in_dim = in_dim
        # print('in_dim: ', self.in_dim)
        self.h_dim = h_dim
        # print('h_dim:', h_dim)
        self.residual = residual

        self.out_dim = out_dim
        self.num_rels = num_rels
        self.final_embed = final_embed
        self.data_file = data_file
        # print('data_file:',data_file)
        self.out_drop = out_drop
        self.num_classes = num_classes
        self.text_item = text_item
        print('text_item:', self.text_item.shape)
        self.img_item = img_item

        self.MMGATLayer = mgat_app_gat_420douban.MMGATLayer
        self.LNet = LNet
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.mmGATdropout = mmGATdropout
        self.linear_hidden_dim = self.h_dim
        # 初始化权重
        self.preference_v = nn.Parameter(nn.init.xavier_normal_(torch.rand((6038, 64))))  # 修改
        self.preference_t = nn.Parameter(nn.init.xavier_normal_(torch.rand((6038, 64))))  # 修改

        self.build_model()
        self.linear = nn.Sequential(
            nn.Linear(64 * 3, 64, bias=False),
        )
        self.linear1 = nn.Parameter(init.xavier_uniform_(torch.rand(128, 64)))
        self.linear2 = nn.Parameter(init.xavier_uniform_(torch.rand(2048, 64)))

    def build_model(self):
        self.layers_Text = nn.Sequential(
            self.MMGATLayer(self.in_dim, self.h_dim,
                            activation=nn.LeakyReLU(negative_slope=0.2), dropout=self.mmGATdropout),
        )
        self.layers_Img = nn.Sequential(
            self.MMGATLayer(self.in_dim, self.h_dim,
                            activation=nn.LeakyReLU(negative_slope=0.2), dropout=self.mmGATdropout),
        )
        self.layers_Id = nn.Sequential(
            self.LNet()
        )
        self.attention_weight = nn.Linear(3, 3, bias=False)
        init.xavier_uniform_(self.attention_weight.weight)
        self.attention_weight_nn = nn.Linear(4, 1, bias=False)

        init.xavier_uniform_(self.attention_weight_nn.weight)

    def forward(self, g):
        # print(11)
        self.text_item.cuda()
        self.img_item.cuda()
        text = torch.cat((self.preference_t, torch.matmul(self.text_item, self.linear1)), dim=0)
        # print(text.shape)
        g.srcdata['h'] = text
        h_, text_attention = self.layers_Text(g)

        img = torch.cat((self.preference_v, torch.matmul(self.img_item, self.linear2)), dim=0)
        g.srcdata['h'] = img
        h_img_, img_attention = self.layers_Img(g)
        # print(img_attention)
        embs_id, attention_value = self.layers_Id(g)

        h_id_ = embs_id

        h = torch.cat((h_, h_img_, h_id_), dim=1)
        # h =  self.batchNorm1(h)
        #
        # h  =self.linear_out(h)
        h = self.linear(h)
        return h, attention_value
