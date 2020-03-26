import torch
import os.path as osp

from data.dataset import FB15Dataset
from dataloader import DataLoader
from model import KBNet


def save_embeddings(run_dir, dataset):
    h_size = 200
    o_size = 200
    heads = 2
    margin = 1
    dev = 'cuda'

    data_loader = DataLoader(dataset)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, dev)

    # determine input size
    x_size = x.shape[1]
    g_size = g.shape[1]

    encoder = KBNet(x_size, g_size, h_size, o_size, heads, margin, device=dev)
    path = osp.join(run_dir, 'encoder.pt')
    model_dict = torch.load(path)
    encoder.load_state_dict(model_dict['model_state_dict'])

    encoder.eval()
    with torch.no_grad():
        h, g = encoder(x, g, edge_idx, edge_type)
    dataset.save_embedding(h, g)


fb = FB15Dataset()
run_dir = './wandb/run-20200324_163404-ozrio67g/'
save_embeddings(run_dir, fb)
