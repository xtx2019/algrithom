import numpy as np
if __name__ == '__main__':
   train_data= np.load("feature_item_movielens.npy",allow_pickle=True)
   print(train_data)
   # train = []
   # for i ,key in enumerate(train_data.keys()):
   #     for item in train_data[key]:
   #         train.append([key,item])
   #         pass
   # np.save("train_list_movielens.npy",train)