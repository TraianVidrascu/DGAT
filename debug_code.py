import time
from functools import reduce
import concurrent.futures
import numpy as np
import pandas as pd
import torch
import sklearn as sk
from sklearn.metrics import euclidean_distances
from sqlalchemy import create_engine

from data.dataset import FB15Dataset
from dataloader import DataLoader

valid_triples = np.random.randint(0, 1500, size=(272115, 2))
valid_types = np.random.randint(0, 237, size=(272115,))
valid_triples = np.stack([valid_triples[:, 0], valid_types, valid_triples[:, 1]])


def is_in(triplet):
    truth = triplet.tolist() in valid_triples.transpose().tolist()
    print(truth)
    return truth


if __name__ == '__main__':
    dataset = FB15Dataset()
    valid_triples = dataset.get_valid_triplets()
    dataset.save_invalid_sampling(valid_triples)
