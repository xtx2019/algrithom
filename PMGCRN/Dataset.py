import time
import random
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class TrainingDataset(Dataset):
    def __init__(self, num_user, num_item, user_item_dict, edge_index):
        self.edge_index = edge_index  # 边下标
        print('edge_index', len(edge_index))
        self.num_user = num_user  # 用户数
        self.num_item = num_item  # 项目数
        self.user_item_dict = user_item_dict  # 用户-项目字典
        self.all_set = set(range(num_user, num_user + num_item))

    def __len__(self):
        return len(self.edge_index)

    def __getitem__(self, index):
        user, pos_item = self.edge_index[index]
        while True:
            neg_item = random.sample(self.all_set, 1)[0]
            if neg_item not in self.user_item_dict[user]:
                break
        return torch.LongTensor([user, user]), torch.LongTensor([pos_item, neg_item])
