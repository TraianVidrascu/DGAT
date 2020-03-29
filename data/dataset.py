import torch
import pickle as pk
import os.path as osp
import networkx as nk
import wandb
import pandas as pd
import numpy as np

from dataloader import DataLoader


class Dataset:
    def __init__(self, raw_dir, processed_dir, eval_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.eval_dir = eval_dir
        self.n = -1
        self.k = -1

    def _load(self):
        with open(self.processed_dir + 'node_map.pickle', 'rb') as handler:
            self.node_mapper = pk.load(handler)
        with open(self.processed_dir + 'rel_map.pickle', 'rb') as handler:
            self.relation_mapper = pk.load(handler)

    def _read_map(self, name):
        file = open(self.raw_dir + name, 'r')
        mapper = {}
        for line in file:
            line = line.strip('\n').split('\t')
            key = line[0]
            idx = int(line[1])
            mapper[key] = idx
        file.close()
        return mapper

    def _read_features(self, name):
        features = []
        file = open(self.raw_dir + name, 'r')
        for line in file:
            row = []
            line = line.strip('\n').split('\t')
            for i, col in enumerate(line):
                if len(col) > 0:
                    row.append(float(col))

            features.append(row)
        x = torch.tensor(features)
        return x

    def _read_edges(self, fold, node_mapper, rel_mapper):
        file = open(self.raw_dir + fold)
        edges = []
        for line in file:
            line = line.strip('\n').split('\t')
            x = node_mapper[line[0]]
            r = rel_mapper[line[1]]
            y = node_mapper[line[2]]

            edge = [x, y, {'label': r}]
            edges.append(edge)
        return edges

    def _edge2graph(self, edges):
        graph = nk.from_edgelist(edges, create_using=nk.MultiDiGraph())
        return graph

    def load_features(self):
        features = torch.load(self.processed_dir + 'features.pt')
        return features

    def load_relations(self):
        relations = torch.load(self.processed_dir + 'relations.pt')
        return relations

    def load_graph(self, fold):
        with open(self.processed_dir + 'graph_' + fold + '.pickle', 'rb') as handle:
            graph = pk.load(handle)
        return graph

    def load_fold(self, fold, dev):
        x = self.load_features().to(dev)
        g = self.load_relations().to(dev)
        graph = self.load_graph(fold)
        return x, g, graph

    def load_embedding(self, dev='cpu'):
        path_h = osp.join(self.processed_dir, 'h.pt')
        path_g = osp.join(self.processed_dir, 'g.pt')
        h = torch.load(path_h).to(dev)
        g = torch.load(path_g).to(dev)
        return h, g

    def save_embedding(self, h, g):
        path_h = osp.join(self.processed_dir, 'h.pt')
        path_g = osp.join(self.processed_dir, 'g.pt')
        torch.save(h, path_h)
        torch.save(g, path_g)
        wandb.save(path_h)
        wandb.save(path_g)

    def read_entities(self):
        mapper = self._read_map('entity2id.txt')
        x = self._read_features('entity2vec.txt')

        torch.save(x, self.processed_dir + 'features.pt')
        with open(self.processed_dir + 'node_map.pickle', 'wb') as handler:
            pk.dump(mapper, handler, protocol=pk.HIGHEST_PROTOCOL)
        return x, mapper

    def read_relations(self):
        mapper = self._read_map('relation2id.txt')
        x = self._read_features('relation2vec.txt')

        torch.save(x, self.processed_dir + 'relations.pt')
        with open(self.processed_dir + 'rel_map.pickle', 'wb') as handler:
            pk.dump(mapper, handler, protocol=pk.HIGHEST_PROTOCOL)
        return x, mapper

    def read_edges_fold(self, fold, node_mapper, rel_mapper):
        edges = self._read_edges(fold + '.txt', node_mapper, rel_mapper)
        graph = self._edge2graph(edges)
        with open(self.processed_dir + 'graph_' + fold + '.pickle', 'wb') as handle:
            pk.dump(graph, handle, protocol=pk.HIGHEST_PROTOCOL)

    def read_edges(self, node_mapper, rel_mapper):
        self.read_edges_fold('train', node_mapper, rel_mapper)
        self.read_edges_fold('valid', node_mapper, rel_mapper)
        self.read_edges_fold('test', node_mapper, rel_mapper)

    def pre_process(self):
        x, node_mapper = self.read_entities()
        g, rel_mapper = self.read_relations()
        self.read_edges(node_mapper, rel_mapper)
        self.save_triplets_raw()

    def save_evaluation_triplets_raw(self, fold, head=True, dev='cpu'):
        x, g, graph = self.load_fold(fold, dev)
        edge_idx, edge_type = DataLoader.graph2idx(graph, dev)

        n = x.shape[0]
        m = edge_idx.shape[1]

        triplets_all = torch.zeros(m, n).long()
        lists_info = torch.zeros(3, m).long()
        for i in range(m):
            triplet_idx = edge_idx[:, i]
            # triplets edge index and valid tripler coordinate
            triplets, position = DataLoader.corrupt_triplet(n, triplet_idx, head=head)

            # corrupted triples
            if head:
                corrupted = triplets[0, :]
                original = triplet_idx[1]
            else:
                corrupted = triplets[1, :]
                original = triplet_idx[0]

            # triplets type, valid position and uncorrupted part
            lists_info[0, i] = original
            lists_info[1, i] = position
            lists_info[2, i] = edge_type[i]

            # append list to all lists
            triplets_all[i, :] = corrupted
            print(i)

        prefix = 'head' if head else 'tail'

        name_triples = prefix + '_' + fold + '_triplets_raw.pt'
        name_lists = prefix + '_' + fold + '_lists_raw.pt'

        torch.save(triplets_all, self.eval_dir + name_triples)
        torch.save(lists_info, self.eval_dir + name_lists)

    def save_triplets_raw(self):
        self.save_evaluation_triplets_raw(fold='valid', head=True)
        self.save_evaluation_triplets_raw(fold='valid', head=False)
        self.save_evaluation_triplets_raw(fold='test', head=True)
        self.save_evaluation_triplets_raw(fold='test', head=False)

    def load_evaluation_triplets_raw(self, fold, head=True, dev='cpu'):
        prefix = 'head' if head else 'tail'
        triplets_path = self.eval_dir + prefix + '_' + fold + '_triplets_raw.pt'
        lists_path = self.eval_dir + prefix + '_' + fold + '_lists_raw.pt'

        triplets = torch.load(triplets_path).to(dev)
        lists = torch.load(lists_path).to(dev)

        return triplets, lists


class FB15Dataset(Dataset):
    def __init__(self):
        super().__init__('./data/FB15k-237/raw/', './data/FB15k-237/processed/', './data/FB15k-237/evaluation/')
        self.n = 14541
        self.k = 237
        self.size_x = 100
        self.size_g = 100
        self.name = 'FB15K-237'


class WN18RR(Dataset):
    def __init__(self):
        super().__init__('./data/WN18RR/raw/', './data/WN18RR/processed/', './data/WN18RR/evaluation/')
        self.n = 40943
        self.k = 11
        self.size_x = 50
        self.size_g = 50
        self.name = 'WN18RR'
