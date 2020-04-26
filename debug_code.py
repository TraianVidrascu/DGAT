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
    dataset = Kinship()
    data_loader = DataLoader(dataset)
    valid_triplets = dataset.get_valid_triplets()
    dataset.save_invalid_sampling(valid_triplets)
    head, tail = dataset.load_invalid_sampling('train')
    _, _, train_graph = data_loader.load_train()
    edge_idx, edge_type = data_loader.graph2idx(train_graph)
    overall = True

    perm = torch.randperm(edge_idx.shape[1])
    edge_idx = edge_idx[:, perm]
    head = head[perm]
    for i, elem in enumerate(head):
        condition = edge_idx[0, i].item() in elem
        print(str(i) + ' ' + str(elem) + ' ' + str(edge_idx[0, i].item()) + ' ' + str(condition))
        overall = overall and condition

    print(overall)
