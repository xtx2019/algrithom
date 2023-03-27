import datetime
import numpy as np
import torch
import math

class EarlyStopping(object):
    def __init__(self, patience=30):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <=self.best_loss) and (acc >= self.best_acc):
                print(loss)
            # if (acc >= self.best_acc):
                self.save_checkpoint(model)
                print(loss)
                print(self.best_loss)
                self.best_loss = np.min((loss, self.best_loss))
                self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))
def full_accuracy(result, num_user,user_item_dict,val_data, step=2000, topk=10):
    user_tensor =result[:num_user]
    item_tensor = result[num_user:]

    start_index = 0
    end_index = num_user if step else num_user

    all_index_of_rank_list = torch.LongTensor([])
    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        print(score_matrix.shape)
        for row, col in user_item_dict.items():
            if row >= start_index and row < end_index:
                row -= start_index
                col = torch.LongTensor(list(col)) - num_user
                # print(row,col)
                score_matrix[row][col] = 1e-5

        _, index_of_rank_list = torch.topk(score_matrix, topk)
        all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu() + num_user), dim=0)
        start_index = end_index
        #
        if end_index + step < num_user:
            end_index += step
        else:
            end_index = num_user

    length = 0
    precision = recall = ndcg = 0.0

    for i,key in enumerate(val_data.keys()):
        user = key
        pos_items = set(val_data[key])
        num_pos = len(pos_items)
        if num_pos == 0:
            continue
        length += 1
        items_list = all_index_of_rank_list[user].tolist()

        items = set(items_list)
        # print(items)
        num_hit = len(pos_items.intersection(items))
        # print(num_hit)
        precision += float(num_hit / topk)
        recall += float(num_hit / num_pos)

        ndcg_score = 0.0
        max_ndcg_score = 0.0

        for i in range(min(num_pos, topk)):
            max_ndcg_score += 1 / math.log2(i + 2)
        if max_ndcg_score == 0:
            continue

        for i, temp_item in enumerate(items_list):
            if temp_item in pos_items:
                ndcg_score += 1 / math.log2(i + 2)

        ndcg += ndcg_score / max_ndcg_score



    return precision / length, recall / length, ndcg / length
