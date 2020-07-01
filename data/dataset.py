import queue
import time
from collections import defaultdict

import torch
import pickle as pk
import os.path as osp
import networkx as nk
import wandb
import pandas as pd
import numpy as np
import functools

from dataloader import DataLoader

BEGINLIST = 'BEGIN'

ENDLIST = 'END'

WN18 = 'WN18RR'
FB15 = 'FB15K-237'
KINSHIP = 'Kinship'
IVY141 = 'Ivy141'


class Dataset:
    def __init__(self, raw_dir, processed_dir, eval_dir, run_dir):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.eval_dir = eval_dir
        self.run_dir = run_dir
        self.n = -1
        self.k = -1
        self.paths = []

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

    def load_mapper(self):
        file = open(self.raw_dir + 'relation2id.txt', 'r')
        mapper = {}
        for line in file:
            line = line.strip('\n').split('\t')
            idx = line[0]
            key = int(line[1])
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

    def load_embedding(self, model_name, dev='cpu'):
        model_name = model_name.lower()
        path_h = osp.join(self.run_dir, 'h_' + model_name + '.pt')
        path_g = osp.join(self.run_dir, 'g_' + model_name + '.pt')
        h = torch.load(path_h).to(dev)
        g = torch.load(path_g).to(dev)
        wandb.save(path_h)
        wandb.save(path_g)
        return h, g

    def save_embedding(self, h, g, model_name):
        path_h = osp.join(self.processed_dir, 'h_' + model_name + '.pt')
        path_g = osp.join(self.processed_dir, 'g_' + model_name + '.pt')
        torch.save(h, path_h)
        torch.save(g, path_g)
        wandb.save(path_h)
        wandb.save(path_g)
        path_h = osp.join(wandb.run.dir, 'h_' + model_name + '.pt')
        path_g = osp.join(wandb.run.dir, 'g_' + model_name + '.pt')
        torch.save(h, path_h)
        torch.save(g, path_g)

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
        # self.save_paths()
        self.filter_evaluation_folds()
        valid_triples = self.get_valid_triplets()
        self.save_invalid_sampling(valid_triples)

    @staticmethod
    def find(tensor, values):
        return torch.nonzero(tensor[..., None] == values)

    def filter_evaluation_triplets(self, fold, head=True):
        x, _, graph = self.load_fold(fold, 'cpu')
        n = x.shape[0]
        edge_idx, edge_type = DataLoader.graph2idx(graph)

        prefix = 'head_' if head else 'tail_'
        triplets_file_path = self.eval_dir + prefix + fold + '_triplets_filtered.txt'
        file = open(triplets_file_path, 'w')

        no_lists = edge_idx.shape[1]
        correct_axis = int(head)
        corrupt_axis = 1 - correct_axis

        correct_triplets = self.get_valid_triplets()
        correct_idx = torch.stack([correct_triplets[0, :], correct_triplets[2, :]])
        correct_type = correct_triplets[1, :]
        for list_idx in range(no_lists):
            correct_part = edge_idx[correct_axis, list_idx]
            triplet_type = edge_type[list_idx]

            # get existing edges with same type and correct end
            same_correct_part = correct_idx[correct_axis, :] == correct_part  # same correct part
            same_correct_type = correct_type[:] == triplet_type  # same type
            same_correct = same_correct_part & same_correct_type

            # generate corrupted and filter existing tuples
            corrupted = torch.randperm(n)
            same = Dataset.find(corrupted, correct_idx[corrupt_axis, same_correct])[:, 0]
            corrupted = torch.LongTensor(np.delete(corrupted.data.numpy(), same))

            # get correct triplet
            correct_triplet = torch.tensor([correct_part, triplet_type, edge_idx[corrupt_axis, list_idx]])

            string_list = BEGINLIST + '\n' + str(correct_triplet.tolist()) + '\n' + str(corrupted[1:].tolist()) + '\n'
            file.write(string_list)
            print(prefix + fold + ' Finished list: %.d' % list_idx)
            file.write(ENDLIST + '\n')
        file.close()

    def filter_evaluation_folds(self):
        self.filter_evaluation_triplets('valid', True)
        self.filter_evaluation_triplets('valid', False)
        self.filter_evaluation_triplets('test', True)
        self.filter_evaluation_triplets('test', False)

    def get_filtered_eval_file(self, fold, head):
        prefix = 'head_' if head else 'tail_'
        triplets_file_path = self.eval_dir + prefix + fold + '_triplets_filtered.txt'
        file = open(triplets_file_path, 'r')
        return file

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
            aux = triplets[:, 0].clone()
            triplets[:, 0] = triplets[:, position].clone()
            triplets[:, position] = aux
            # corrupted triples
            if head:
                corrupted = triplets[0, :]
                original = triplet_idx[1]
            else:
                corrupted = triplets[1, :]
                original = triplet_idx[0]

            # triplets type, valid position and uncorrupted part
            lists_info[0, i] = original
            lists_info[1, i] = 0
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

    def load_paths(self, use_partial):
        if use_partial:
            path_name = self.processed_dir + 'partial.pt'
        else:
            path_name = self.processed_dir + 'paths.pt'
        paths = torch.load(path_name)
        return paths

    def bfs(self, graph, node):
        visit = {}
        distance = {}
        parent = {}
        distance_lengths = {}

        visit[node] = 1
        distance[node] = 0
        parent[node] = (-1, -1)

        q = queue.Queue()
        q.put((node, -1))

        while not q.empty():
            top = q.get()
            if top[0] in graph.nodes():
                for target in graph[top[0]].keys():
                    if (target in visit.keys()):
                        continue
                    else:
                        q.put((target, graph[top[0]][target][0]['label']))

                        distance[target] = distance[top[0]] + 1

                        visit[target] = 1
                        if distance[target] > 2:
                            continue
                        parent[target] = (top[0], graph[top[0]][target][0]['label'])

                        if distance[target] not in distance_lengths.keys():
                            distance_lengths[distance[target]] = 1
        neighbors = {}
        for target in visit.keys():
            if (distance[target] != 2):
                continue
            relations = []
            entities = [target]
            temp = target
            while (parent[temp] != (-1, -1)):
                relations.append(parent[temp][1])
                entities.append(parent[temp][0])
                temp = parent[temp][0]

            if (distance[target] in neighbors.keys()):
                neighbors[distance[target]].append(
                    (tuple(relations), tuple(entities[:-1])))
            else:
                neighbors[distance[target]] = [
                    (tuple(relations), tuple(entities[:-1]))]

        return neighbors

    def get_further_neighbors(self):
        _, _, graph = self.load_fold('train', 'cpu')
        nodes = graph.nodes()
        neighbors = {}
        start_time = time.time()
        print("length of graph keys is ", len(nodes))
        for source in nodes:
            temp_neighbors = self.bfs(graph, source)
            for distance in temp_neighbors.keys():
                if (source in neighbors.keys()):
                    if (distance in neighbors[source].keys()):
                        neighbors[source][distance].append(
                            temp_neighbors[distance])
                    else:
                        neighbors[source][distance] = temp_neighbors[distance]
                else:
                    neighbors[source] = {}
                    neighbors[source][distance] = temp_neighbors[distance]
            print('Done: %.d' % source)
        print("time taken ", time.time() - start_time)

        print("length of neighbors dict is ", len(neighbors))
        return neighbors

    def save_paths(self):
        neighbors = self.get_further_neighbors()
        path_triplets = []
        partial_paths = []
        for key in neighbors.keys():
            paths = neighbors[key]
            for i, path in enumerate(paths[2]):
                x = key
                g_1 = path[0][0]
                g_2 = path[0][1]
                y = path[1][0]

                path_triplets.append(torch.tensor([x, g_1, g_2, y]).long())
                if i < 2:
                    partial_paths.append(torch.tensor([x, g_1, g_2, y]).long())

        path_triplets = torch.stack(path_triplets)
        file = self.processed_dir + 'paths.pt'
        torch.save(path_triplets, file, pickle_protocol=pk.HIGHEST_PROTOCOL)

        partial_paths = torch.stack(partial_paths)
        file = self.processed_dir + 'partial.pt'
        torch.save(partial_paths, file, pickle_protocol=pk.HIGHEST_PROTOCOL)

    @staticmethod
    def dd():
        return defaultdict(set)

    def _merge_type_and_ends(self, edge_idx, edge_type):
        row, col = edge_idx
        triplet = torch.stack([row, edge_type, col])
        return triplet

    def get_valid_triplets(self):
        _, _, graph_train = self.load_fold('train', 'cpu')
        _, _, graph_valid = self.load_fold('valid', 'cpu')
        _, _, graph_test = self.load_fold('test', 'cpu')

        train_idx, train_type = DataLoader.graph2idx(graph_train)
        triplets_train = self._merge_type_and_ends(train_idx, train_type)
        valid_idx, valid_type = DataLoader.graph2idx(graph_valid)
        triplets_valid = self._merge_type_and_ends(valid_idx, valid_type)
        test_idx, test_type = DataLoader.graph2idx(graph_test)
        triplets_test = self._merge_type_and_ends(test_idx, test_type)

        valid_triplets = torch.cat([triplets_train, triplets_valid, triplets_test], dim=1)
        return valid_triplets

    def save_invalid_sampling(self, valid_triplets):
        head_map = defaultdict(self.dd)
        tail_map = defaultdict(self.dd)
        for i, triplet in enumerate(valid_triplets.t()):
            row, rel, col = triplet.long()
            row, rel, col = row.item(), rel.item(), col.item()
            # head part
            head_map[col][rel].add(row)
            # tail part
            tail_map[row][rel].add(col)
            print('Processed %.d' % i)

        _, _, train_graph = self.load_fold('train', 'cpu')
        _, _, graph_valid = self.load_fold('valid', 'cpu')
        _, _, graph_test = self.load_fold('test', 'cpu')

        train_idx, train_type = DataLoader.graph2idx(train_graph)
        valid_idx, valid_type = DataLoader.graph2idx(graph_valid)
        test_idx, test_type = DataLoader.graph2idx(graph_test)

        self._save_invalid_sampling_fold(head_map, tail_map, train_idx, train_type, 'train')
        self._save_invalid_sampling_fold(head_map, tail_map, valid_idx, valid_type, 'valid')
        self._save_invalid_sampling_fold(head_map, tail_map, test_idx, test_type, 'test')

    def _save_invalid_sampling_fold(self, head_map, tail_map, edge_idx, edge_type, fold):
        head_invalid_sampling = []
        tail_invalid_sampling = []
        for i in range(edge_idx.shape[1]):
            row, col = edge_idx[:, i].long()
            rel = edge_type[i].long()
            row, rel, col = row.item(), rel.item(), col.item()

            # head part
            invalid_heads = head_map[col][rel]
            head_invalid_sampling.append(invalid_heads)

            # tail part
            invalid_tails = tail_map[row][rel]
            tail_invalid_sampling.append(invalid_tails)
            print('Processed %.d' % i + ' biggest: %.d' % max(len(invalid_tails), len(invalid_heads)))
        head_path = self.processed_dir + fold + '_head_sampling.pk'
        tail_path = self.processed_dir + fold + '_tail_sampling.pk'
        with open(head_path, 'wb') as handler:
            head_invalid_sampling = np.array(head_invalid_sampling)
            pk.dump(head_invalid_sampling, handler, pk.HIGHEST_PROTOCOL)
        with open(tail_path, 'wb') as handler:
            tail_invalid_sampling = np.array(tail_invalid_sampling)
            pk.dump(tail_invalid_sampling, handler, pk.HIGHEST_PROTOCOL)

    def load_invalid_sampling(self, fold='train'):
        head_path = self.processed_dir + fold + '_head_sampling.pk'
        tail_path = self.processed_dir + fold + '_tail_sampling.pk'
        with open(head_path, 'rb') as handler:
            head_invalid_sampling = pk.load(handler)
        with open(tail_path, 'rb') as handler:
            tail_invalid_sampling = pk.load(handler)
        return head_invalid_sampling, tail_invalid_sampling


class FB15Dataset(Dataset):
    def __init__(self):
        super().__init__('./data/FB15k-237/raw/', './data/FB15k-237/processed/', './data/FB15k-237/evaluation/',
                         './data/FB15k-237/run/')
        self.n = 14541
        self.k = 237
        self.size_x = 100
        self.size_g = 100
        self.name = FB15


class WN18RR(Dataset):
    def __init__(self):
        super().__init__('./data/WN18RR/raw/', './data/WN18RR/processed/', './data/WN18RR/evaluation/',
                         './data/WN18RR/run/')
        self.n = 40943
        self.k = 11
        self.size_x = 50
        self.size_g = 50
        self.name = WN18


class Kinship(Dataset):
    def __init__(self):
        super().__init__('./data/kinship/raw/', './data/kinship/processed/', './data/kinship/evaluation/',
                         './data/kinship/run/')
        self.n = 104
        self.k = 25
        self.size_x = 200
        self.size_g = 200
        self.name = KINSHIP


class Ivy(Dataset):
    def __init__(self):
        super().__init__('./data/ivy141-all/raw/', './data/ivy141-all/processed/', './data/ivy141-all/evaluation/',
                         './data/ivy141-all/run/')
        self.n = 9738
        self.k = 21
        self.size_x = 100
        self.size_g = 150
        self.name = IVY141
