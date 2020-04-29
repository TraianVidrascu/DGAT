import time
from functools import reduce
import concurrent.futures
import numpy as np
import pandas as pd
import torch

import sklearn as sk
from sklearn.metrics import euclidean_distances
from sqlalchemy import create_engine

from data.dataset import FB15Dataset, WN18RR, Kinship
from dataloader import DataLoader

if __name__ == '__main__':

    fold, head = 'valid', False
    dataset = Kinship()
    # dataset.pre_process()
    dev = 'cuda'
    data_loader = DataLoader(dataset)
    triplets_file = data_loader.get_filtered_eval_file(fold, head)
    valid_triplets = dataset.get_valid_triplets().t().cuda()
    valid_triplets = valid_triplets.tolist()
    valid_triplets = list(map(lambda x: tuple(x), valid_triplets))
    while True:
        edge_idx, edge_type, position = DataLoader.load_list(triplets_file, head, dev)

        # if no more lists break
        if edge_idx is None:
            break
        res = 0
        for i in range(0, edge_idx.shape[1]):
            current_triplet = torch.stack([edge_idx[0, i], edge_type[i], edge_idx[1, i]])
            cond = tuple(current_triplet[:].tolist()) in valid_triplets
            if cond:
                res += 1
            else:
                continue
        print(res)
