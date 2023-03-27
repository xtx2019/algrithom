import os
import pickle
import random

import numpy
import numpy as np
import torch


#  计算情感得分
def compute_sentiment():
    sentiment_file = "item2sentiment.pickle"
    item2sentiment_ = {}
    with open(sentiment_file, 'rb') as file:
        item2sentiment = pickle.load(file)
        sum = 0.0

        for i_, s_list in item2sentiment.items():
            temp_list = []
            len_1 = len(s_list)
            for s in s_list:
                if s == 'positive':
                    temp_list.append(1.0)
                else:
                    temp_list.append(0.0)
            len_2 = len(temp_list)
            if len_1 != len_2:
                print('something wrong!!')
            item2sentiment_[i_] = np.mean(temp_list) ** 0.1
            sum += item2sentiment_[i_]
        f = open('sentiment.txt', 'w')
        for i, s in item2sentiment_.items():
            item2sentiment_[i] = item2sentiment_[i] / sum * len(item2sentiment_)
            f.write(str(s)+'\n')
        f.close()
    file.close()
    f = open('sentiment.txt','r')
    list = []
    for i in f.readlines():
        list.append(float(i.strip('\n')))
    list = torch.Tensor(list)
    print(list)
print(random.uniform(0.9, 1))

#compute_sentiment()
