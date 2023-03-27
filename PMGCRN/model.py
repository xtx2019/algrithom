import torch.nn as nn
import torch
import mgat_app
from functools import partial
import torch.nn.functional as F
import numpy as np
import os
import time
from torch.nn import init


class Model(nn.Module):
    def __init__(self, num_nodes, in_dim, h_dim, out_dim, num_classes, num_rels,
                 num_hidden_layers=0, dropout=0.35, mmGATdropout=0.35, out_drop=0.65, residual=0.12, gat_type='sum',
                 data_file='result/', final_embed='IMDB_SUM', text_item=None, img_item=None):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.residual = residual
        self.out_dim = out_dim
        self.num_rels = num_rels  # 关系的数量
        self.final_embed = final_embed
        self.data_file = data_file
        self.out_drop = out_drop
        self.num_classes = num_classes
        self.text_item = text_item
        self.img_item = img_item
        # if gat_type =='sum':
        #     self.MMGATLayer = mgat_pat_sum.MMGATLayer
        # else:
        #     print(1)
        #     self.MMGATLayer = mgat_pat_max.MMGATLayer

        self.MMGATLayer = mgat_app.MMGATLayer

        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.mmGATdropout = mmGATdropout
        # 隐藏层即全连接层的(神经元)的个数
        self.linear_hidden_dim = self.h_dim
        # 初始化项目的特征向量
        self.preference_v = nn.init.xavier_normal_(torch.rand((6038, 64), requires_grad=True)).cuda()
        self.preference_t = nn.init.xavier_normal_(torch.rand((6038, 64), requires_grad=True)).cuda()
        self.build_model()
        self.linear = nn.Sequential(
            nn.Linear(self.linear_hidden_dim * (1 + self.num_hidden_layers), self.out_dim, bias=False),
            nn.Dropout(self.out_drop),
            nn.LeakyReLU()
        )

        self.linear1 = nn.Parameter(torch.Tensor(128, 64))
        self.linear2 = nn.Parameter(torch.Tensor(2048, 64))

        init.xavier_uniform_(self.linear1)
        init.xavier_uniform_(self.linear2)

        self.batchNorm1 = nn.Sequential(
            nn.BatchNorm1d(num_features=(self.out_dim * 2)),
            # nn.Tanh()
        )

    def build_model(self):
        self.layers = nn.ModuleList()

        i2h = self.build_input_layer()
        self.layers.append(i2h)

        for _ in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)

    def build_input_layer(self):

        linear = nn.Sequential(
            self.MMGATLayer(self.in_dim, self.h_dim, self.num_rels,
                            activation=F.relu, dropout=self.mmGATdropout, residual=self.residual, norm='both',
                            weight=True),
        )

        return linear

    def build_hidden_layer(self):

        linear = nn.Sequential(
            self.MMGATLayer(self.h_dim, self.h_dim, self.num_rels,
                            activation=F.relu, dropout=self.mmGATdropout, residual=self.residual, norm='both',
                            weight=True)

        )
        return linear

    def forward(self, g):
        print("self.preference", self.preference_t)
        # print(11)
        self.text_item.cuda()
        # self.img_item.cuda()
        text = torch.cat((self.preference_t, torch.matmul(self.text_item, self.linear1)), dim=0)
        g.srcdata['h'] = text
        h = []
        # print(len([g,text]))
        # print(self.layers[0])
        text = self.layers[0](g)
        # print(text.shape)
        h.append(text)
        for layer in self.layers[1:]:
            g.srcdata['h'] = text
            text = layer(g)
            h.append(text)
        h_ = torch.empty_like(h[0]).cuda()
        for h_t in h:
            h_ = torch.add(h_, h_t)
        # print(h_.shape)
        img = torch.cat((self.preference_v, torch.matmul(self.img_item, self.linear2)), dim=0)
        g.srcdata['h'] = img
        h_img = []
        # print(len([g, text]))
        # print(self.layers[0])
        img = self.layers[0](g)
        # print(text.shape)
        h_img.append(img)
        for layer in self.layers[1:]:
            g.srcdata['h'] = img
            img = layer(g)
            h_img.append(img)
        h_img_ = torch.empty_like(h_img[0]).cuda()
        for h_i in h_img:
            h_img_ = torch.add(h_img_, h_i)
        # print(h_img_.shape)
        h = torch.cat((h_, h_img_), dim=1)
        h = self.batchNorm1(h)
        #
        # h  =self.linear_out(h)
        return h

    def embedding_out(self, g):
        g.ndata['h'] = (g.ndata['f'])
        g.ndata['h_img'] = (g.ndata['f_img'])
        h = []

        t = self.layers[0](g)
        t_text = g.ndata['h']
        t_img = g.ndata['h_img']
        print(t.shape)
        h.append(t)
        for layer in self.layers[1:]:
            h_layer = layer(g)
            # g.ndata['h'] =t_text+g.ndata['h']
            # g.ndata['h_img'] = t_img+g.ndata['h_img']

            h.append(h_layer)
        h = torch.cat(h, dim=1)
        h = self.linear(h)
        h = h.cpu().detach().numpy()

        self.save_result(self.final_embed, h, self.data_file)
# def save_result(self,t, final_embed, data_file):
#     att_output_files = data_file
#     if not os.path.exists(att_output_files):
#         os.makedirs(att_output_files)
#     result_emb = att_output_files + str(t) + ".out"
#     f = open(result_emb, 'w', encoding='utf-8')
#     for i in range(np.shape(final_embed)[0]):
#         f.write(str(i) + ' ')
#         # print(vocab[i])
#         for j in range(np.shape(final_embed)[1]):
#             f.write(str(final_embed[i, j].real) + ' ')
#         f.write('\n')
#     f.close()
