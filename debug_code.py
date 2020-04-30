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


def save_embed():
    file = './eval_dir/embeddings/trained_3599.pth'
    model = torch.load(file)
    h = model['final_entity_embeddings']
    g = model['final_relation_embeddings']
    h_path = './eval_dir/embeddings/h_kbat_kinship.pt'
    g_path = './eval_dir/embeddings/g_kbat_kinship.pt'

    torch.save(h, h_path)
    torch.save(g, g_path)


if __name__ == '__main__':
    save_embed()