import networkx as nx
import numpy as np
import scipy
import pickle
import dgl
import torch


def load_Movielens_data(prefix='data/movielens'):
    features_1 = np.load(prefix + '/feature_item_movielens.npy')
    # print('features_1:', features_1,features_1.shape)
    # img
    features_1_img = np.load(prefix + '/img_feature_item_movieLens.npz')['feature']
    # print('features_1_img:', features_1_img,features_1_img.shape)
    train_list = np.load(prefix + "/train_list_movielens.npy", allow_pickle=True).tolist()
    val_list = np.load(prefix + "/val_past_movielens.npy", allow_pickle=True).tolist()
    test_list = np.load(prefix + "/test_past_movielens.npy", allow_pickle=True).tolist()
    # print('train_list:', train_list)
    f = open("train_list.txt", "w")
    f.write(str(train_list))
    # train_list= np.load(prefix + train_val_test_dir)
    # np.set_printoptions(threshold=np.inf)  # 完全显示矩阵
    rdf = np.load(prefix + '/users_items_rdf_list.npz')
    # print(rdf['ends'])
    users_items = np.load(prefix + '/train_past_movielens.npy', allow_pickle=True).tolist()
    # (users_items)
    return [features_1], [features_1_img], train_list, val_list, test_list, rdf, users_items


def load_Amazon_data(prefix='data/amazon'):
    features_1 = np.load(prefix + '/feature_item_amazon.npy')

    # img
    features_1_img = np.load(prefix + '/img_feature_amazon.npz')['feature']
    train_list = np.load(prefix + "/train_list_amazon.npy", allow_pickle=True).tolist()
    val_list = np.load(prefix + "/val_amazon.npy", allow_pickle=True).tolist()
    test_list = np.load(prefix + "/test_amazon.npy", allow_pickle=True).tolist()

    # train_list= np.load(prefix + train_val_test_dir)
    rdf = np.load(prefix + '/users_items_rdf_list.npz')
    users_items = np.load(prefix + '/train_amazon.npy', allow_pickle=True).tolist()
    return [features_1], [features_1_img], train_list, val_list, test_list, rdf, users_items


def load_Douban_data(prefix='data/douban'):
    features_1 = np.load(prefix + '/feature_item_douban.npy')
    # print('features_1', features_1)
    # img
    features_1_img = np.load(prefix + '/img_feature_douban.npz')['feature']
    # print('features_1_img', features_1_img)
    train_list = np.load(prefix + "/train_list_douban.npy", allow_pickle=True).tolist()
    # print('train_list', len(train_list))
    val_list = np.load(prefix + "/val_douban.npy", allow_pickle=True).tolist()
    # print('val_list', val_list)
    test_list = np.load(prefix + "/test_douban.npy", allow_pickle=True).tolist()
    # print('test_list', test_list)
    # train_list= np.load(prefix + train_val_test_dir)
    rdf = np.load(prefix + '/users_items_rdf_list.npz')
    users_items = np.load(prefix + '/train_douban.npy', allow_pickle=True).tolist()
    # print('users_items', users_items)
    return [features_1], [features_1_img], train_list, val_list, test_list, rdf, users_items


if __name__ == '__main__':
    prefix = 'data/movielens'
    load_Movielens_data(prefix)
    pass
