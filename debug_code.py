import time
from functools import reduce
import concurrent.futures
import numpy as np
import pandas as pd
import torch

import sklearn as sk
from sklearn.metrics import euclidean_distances
from sqlalchemy import create_engine

from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader

valid_triples = np.random.randint(0, 1500, size=(272115, 2))
valid_types = np.random.randint(0, 237, size=(272115,))
valid_triples = np.stack([valid_triples[:, 0], valid_types, valid_triples[:, 1]])


def is_in(triplet):
    truth = triplet.tolist() in valid_triples.transpose().tolist()
    print(truth)
    return truth


import wandb

if __name__ == '__main__':
    file_a = "eval_dir/check/encoder_kbat_fb15k-237.pt"
    file_b = "eval_dir/check/encoder_kbat_fb15k-237_0.pt"
    file_c = "./eval_dir/check/encoder_kbat_fb15k-237_server.pt"
    a = torch.load(file_a)
    b = torch.load(file_b)
    c = torch.load(file_c)
    a = a['model_state_dict']
    b = b['model_state_dict']
    c = c['model_state_dict']
    for key in a.keys():
        print((torch.sqrt(a[key]-b[key])**2).mean())
        print((torch.sqrt(a[key]-c[key])**2).mean())
