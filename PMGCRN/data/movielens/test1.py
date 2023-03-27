import numpy as np
if __name__ == '__main__':
    user_map=np.load("users_items.npy",allow_pickle=True).item()
    i=0
    for key in user_map:
        i +=len(user_map[key])
    print(i)