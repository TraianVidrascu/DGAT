import torch
import numpy as np
import ast

from data.dataset import FB15Dataset
from dataloader import DataLoader


def load_list(file, head=True, dev='cpu'):
    line = file.readline()
    if line == '':
        return None, None

    array = None, None
    while "END" not in line and line != '':
        array = ast.literal_eval(line)
        line = file.readline()

    if array is not None:
        triplets = torch.tensor(array).to(dev)
        correct = triplets[:, 0]
        corrupted = triplets[:, 2]
        edge_type = triplets[:, 1]
        if head:
            edge_idx = torch.stack([corrupted, correct])
        else:
            edge_idx = torch.stack([correct, corrupted])
        return edge_idx, edge_type

    return None, None


def evaluate_filtered(model, h, g, dataloader, fold, head, dev='cpu'):
    with torch.no_grad():
        # load corrupted head triplets

        triplets_file = dataloader.get_filtered_eval_file(fold, head)
        counter = 0
        while True:
            edge_idx, edge_type = load_list(triplets_file)
            counter += 1
            print(edge_idx.shape)
            if edge_idx is None:
                break
            print(counter)
        triplets_file.close()
        return None


dataset = FB15Dataset()
dataloader = DataLoader(dataset)

evaluate_filtered(None, None, None, dataloader, 'test', True, )
