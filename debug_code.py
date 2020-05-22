import time
from functools import reduce
import concurrent.futures
import numpy as np
import pandas as pd
import pickle as pk
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


def verify_evaluation(data_loader):
    fold = 'test'
    head = False
    file = data_loader.get_filtered_eval_file(fold, head)
    valid_triplets = list(map(lambda x: tuple(x), dataset.get_valid_triplets().t().tolist()))
    idx = 0
    while True:
        edge_idx, edge_type, position = data_loader.load_list(file, head, 'cuda')

        if edge_idx is None:
            break
        eval_triplets = list(
            map(lambda x: tuple(x), torch.stack([edge_idx[0, 1:], edge_type[1:], edge_idx[1, 1:]]).t().tolist()))

        for triplet in eval_triplets:
            res = triplet in valid_triplets
            if res:
                print(triplet, len(eval_triplets))
        idx += 1
        print('done %.d' % idx)
        if position > 0:
            print('pula')
            break


def transfer_embeddings():
    file = './eval_dir/embeddings/trained_3599.pth'
    file_h = './eval_dir/embeddings/h_kbat_kinship.pt'
    file_g = './eval_dir/embeddings/g_kbat_kinship.pt'
    trained = torch.load(file)
    h = torch.load(file_h)
    g = torch.load(file_g)
    trained['final_entity_embeddings'] = h
    trained['final_relation_embeddings'] = g

    file = './eval_dir/embeddings/trained_4000.pth'
    torch.save(trained, file)


def transfer_paper_decoder_2_my_decoder(paper_decoder, decoder):
    decoder['model_state_dict']['node_embeddings'] = paper_decoder['final_entity_embeddings']
    decoder['model_state_dict']['rel_embeddings'] = paper_decoder['final_relation_embeddings']
    decoder['model_state_dict']['conv.conv_layer.weight'] = paper_decoder['convKB.conv_layer.weight']
    decoder['model_state_dict']['conv.conv_layer.bias'] = paper_decoder['convKB.conv_layer.bias']
    decoder['model_state_dict']['conv.fc_layer.weight'] = paper_decoder['convKB.fc_layer.weight']
    decoder['model_state_dict']['conv.fc_layer.bias'] = paper_decoder['convKB.fc_layer.bias']
    file = 'eval_dir/decoder/ConvKB_kbat_kinship.pt'
    torch.save(decoder, file)


if __name__ == '__main__':
    # file = './eval_dir/decoder/trained_399.pth'
    # decoder = torch.load(file)
    # file_2 = 'eval_dir/decoder/temporary.pt'
    # my_decoder = torch.load(file_2)
    # transfer_paper_decoder_2_my_decoder(decoder, my_decoder)
    # print(decoder)
    # print(my_decoder['model_state_dict'])

    # dataset = Kinship()
    # ldr = DataLoader(dataset)
    # verify_evaluation(ldr)

    # file = './data/FB15k-237/2hop.pickle'
    # with open(file, 'rb') as f:
    #     a = pk.load(f)
    # print(a)
    pass
