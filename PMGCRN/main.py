import process
import dgl
import torch
from model_gat_420douban import Model
from sklearn.metrics import f1_score, accuracy_score
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import numpy as np
import numpy.random as random
import os
from utils import EarlyStopping, full_accuracy
from evaluate_own import *
from Dataset import TrainingDataset


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


seed = 0


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = accuracy_score(prediction, labels)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1


def evaluate(model, g, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        logits = model(g)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])
    return loss, accuracy, micro_f1, macro_f1


def load_g(datasets):
    # print('------这里是load_g函数------')
    print("dataset:" + datasets)
    if datasets == 'movielens':
        features_list, features_list_img, train_list, val_list, test_list, rdf, users_items = process.load_Movielens_data()
    if datasets == 'amazon':
        features_list, features_list_img, train_list, val_list, test_list, rdf, users_items = process.load_Amazon_data()
    if datasets == 'douban':
        # 项目特征、项目图片特征、
        features_list, features_list_img, train_list, val_list, test_list, rdf, users_items = process.load_Douban_data()

    features_list = [torch.FloatTensor(features).to(device) for features in features_list]
    features_list_img = [torch.FloatTensor(features).to(device) for features in features_list_img]
    features_all = torch.cat((features_list), dim=0)
    features_img_all = torch.cat((features_list_img), dim=0)
    num_nodes = features_all.shape[0]  # 1507行
    feature_size = features_all.shape[-1]  # 128列
    feature_img_size = features_img_all.shape[-1]
    print("img_size{}:".format(feature_img_size))  # 图像特征个数
    g = dgl.DGLGraph()  # 构造图
    g.add_nodes(9921)  # ????????????????????????????????????????????????????为什么是5868=项目数+用户数 #修改
    start = rdf['starts']
    # np.set_printoptions(threshold=np.inf)   #完全显示矩阵
    end = rdf['ends']
    g.add_edges(start, end)
    g = g.to(device)

    train_idx = train_list  # 训练集
    val_idx = val_list  # 验证集
    test_idx = test_list  # 测试集
    return g, train_idx, val_idx, test_idx, num_nodes, feature_size, feature_img_size, users_items, features_all, features_img_all


def main(args):
    max_v = -1
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    g, train_idx, val_idx, test_idx, num_nodes, feature_size, feature_img_size, users_items, features_all, features_img_all = load_g(
        args.datasets)
    train_dataset = TrainingDataset(6038, 3883, users_items, train_idx)
    train_dataloader = DataLoader(train_dataset, 1000, shuffle=True)
    # print(g)
    gat_type = args.gat_type
    n_epochs = args.epochs  # epochs to train
    num_classes = args.num_classes
    n_hidden_layers = args.num_layers
    lr = 0.0001
    num_rels = args.num_rels
    dropout = args.in_drop
    mmGATdropout = args.attn_drop
    residual = args.residual
    out_drop = args.out_drop
    out_size = args.out_size

    data_file = args.data_file

    # create model
    model = Model(num_nodes=num_nodes, in_dim=64,
                  h_dim=64,
                  out_dim=out_size,
                  num_classes=num_classes,
                  num_rels=num_rels,
                  num_hidden_layers=n_hidden_layers, dropout=dropout, mmGATdropout=mmGATdropout, out_drop=out_drop,
                  residual=residual, gat_type=gat_type, data_file=data_file, final_embed=None, text_item=features_all,
                  img_item=features_img_all)  # in_dim 表示的是输入的矩阵特征数  out_dim 表示的是输出的矩阵特征数

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    print("start training...")
    dt = datetime.datetime.now()

    precision_t_, recall_t_, ndcg_t_ = 0, 0, 0

    for epoch in range(n_epochs):
        model.train()
        torch.cuda.empty_cache()
        print("*************")
        for user_tensor1, item_tensor1 in tqdm(train_dataloader):
            user_tensor, item_tensor = user_tensor1, item_tensor1
            # continue
            optimizer.zero_grad()
            # print('optimizer:', optimizer)
            out, attention = model.forward(g)
            # print(out,attention)
            user_tensor = user_tensor.view(-1)
            item_tensor = item_tensor.view(-1)
            # print('user_tensor:', user_tensor.shape)
            weight = torch.tensor([[1.0], [-1.0]]).cuda()
            user_score = out[user_tensor]
            item_score = out[item_tensor]

            score = torch.sum(user_score * item_score, dim=1).view(-1, 2)

            score_ = torch.matmul(score, weight)

            loss = -torch.mean(torch.log(torch.sigmoid(score_)))

            loss.backward()
            optimizer.step()

        print('epoch:', epoch)
        print('loss:', loss)
        print("attention{}".format(attention))
        model.eval()
        with torch.no_grad():
            out, attention = model.forward(g)
        precision, recall, ndcg = full_accuracy(out, 6038, users_items, val_data=val_idx, step=2000, topk=10)  # 修改
        precision_t, recall_t, ndcg_t = full_accuracy(out, 6038, users_items, val_data=test_idx, step=2000,
                                                      topk=10)  # 修改

        print(precision, recall, ndcg)
        print(precision_t, recall_t, ndcg_t)

        if recall > max_v:
            max_v = recall
            max_precision = precision
            max_ndcg = ndcg
            precision_t_ = precision_t
            recall_t_ = recall_t
            ndcg_t_ = ndcg_t
            filename = 'movielens_early_stop_{}_{:02d}-{:02d}-{:02d}_all.pth'.format(
                dt.date(), dt.hour, dt.minute, dt.second)  # 修改
            torch.save(model.state_dict(), filename)
        print("max_precision{},max_recall{},max_ndcg{}".format(max_precision, max_v, max_ndcg))
        print("max_precision_t{},max_recall_t{},max_ndcg_t{}".format(precision_t_, recall_t_, ndcg_t_))


if __name__ == '__main__':
    # torch.cuda.empty_cache函数可以清空GPU的缓存，把GPU上已分配的显存清空，这样可以释放出资源，提升GPU的计算性能。
    torch.cuda.empty_cache()
    parser = argparse.ArgumentParser(description='MMRGAT')
    parser.add_argument("--gat-type", type=str, default='max',
                        help="type of gat  divide into sum an max")
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")

    parser.add_argument('--num-classes', type=int, default=3,
                        help="number of clasese")
    parser.add_argument('--num-rels', type=int, default=2,
                        help="number of relation")
    parser.add_argument("--num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--hidden-size", type=int, default=64,
                        help="hidden feature size")
    parser.add_argument("--out-size", type=int, default=64,
                        help="out feature size")
    parser.add_argument("--residual", type=float, default=0.12,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0.6,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0.6,
                        help="attention dropout")
    parser.add_argument("--out-drop", type=float, default=.6,
                        help="linnear out dropout")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--early-stop', action='store_true', default=False,
                        help="indicates whether to use early stop or not")

    parser.add_argument('--datasets', type=str, default='movielens',
                        help="change dataset")  # 修改

    parser.add_argument('--data-file', type=str, default='result/',
                        help="change dataset")

    parser.add_argument('--final-embed', type=str, default='IMDB',
                        help="change dataset")
    args = parser.parse_args()

    main(args)
