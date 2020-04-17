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
        self.save_paths(2)
        self.save_triplets_raw()
        self.filter_evaluation_folds()
        valid_triples = self.get_valid_triplets()
        self.save_invalid_sampling(valid_triples)

    @staticmethod
    def find(tensor, values):
        return torch.nonzero(tensor[..., None] == values)

    def filter_evaluation_triplets(self, fold, head=True):
        triplets_raw, lists_raw = self.load_evaluation_triplets_raw(fold, head)
        if fold == 'valid':
            other_fold = 'test'
        else:
            other_fold = 'valid'

        _, _, graph_other = self.load_fold(other_fold, 'cpu')
        other_idx, other_type = DataLoader.graph2idx(graph_other)

        _, _, graph_train = self.load_fold('train', 'cpu')
        train_idx, train_type = DataLoader.graph2idx(graph_train)

        prefix = 'head_' if head else 'tail_'
        triplets_file_path = self.eval_dir + prefix + fold + '_triplets_filtered.txt'
        file = open(triplets_file_path, 'w')

        no_lists = triplets_raw.shape[0]
        m = triplets_raw.shape[1]
        correct_axis = int(head)
        corrupt_axis = 1 - correct_axis

        for list_idx in range(no_lists):
            # get triplet uncorrupted information
            correct_part = lists_raw[0, list_idx]
            position = lists_raw[1, list_idx]
            triplet_type = lists_raw[2, list_idx]

            # same train set
            same_part_train = train_idx[correct_axis, :] == correct_part  # same correct part
            same_type_train = train_type[:] == triplet_type  # same type
            same_correct_train = same_part_train & same_type_train

            # same other fold
            same_part_other = other_idx[correct_axis, :] == correct_part  # same correct part
            same_type_other = other_type[:] == triplet_type  # same type
            same_correct_other = same_part_other & same_type_other

            corrupted = triplets_raw[list_idx, :]
            correct_triplet = torch.tensor([correct_part, triplet_type, corrupted[position]])

            same_train = Dataset.find(corrupted, train_idx[corrupt_axis, same_correct_train])[:, 0]
            corrupted = torch.LongTensor(np.delete(corrupted.data.numpy(), same_train))

            same_other = Dataset.find(corrupted, other_idx[corrupt_axis, same_correct_other])[:, 0]
            corrupted = torch.LongTensor(np.delete(corrupted.data.numpy(), same_other))

            string_list = BEGINLIST + '\n' + str(correct_triplet.tolist()) + '\n' + str(corrupted.tolist()) + '\n'
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

    def load_paths(self):
        path_name = self.processed_dir + 'paths_2.pkl'
        with open(path_name, 'rb') as handler:
            paths = pk.load(handler)
        return paths

    def dfs_path(self, graph, path, node, depth, max_depth):
        depth += 1
        if depth == 0:
            path = [node]
        if depth == max_depth:
            new_path = path + [node]
            self.paths.append(new_path)
            return
        outbounds = graph[node]
        for outbound in outbounds.keys():
            label = outbounds[outbound][0]['label']
            new_path = path + [label]
            self.dfs_path(graph, new_path, outbound, depth, max_depth)

    def save_paths(self, depth=2):
        _, _, graph = self.load_fold('train', 'cpu')
        nodes = graph.nodes()
        columns = ['x'] + ['g_' + str(i) for i in range(1, depth + 1)] + ['y']

        for node in nodes:
            self.dfs_path(graph, [], node, -1, depth)
            print('Launched: ' + str(node) + ' depth: ' + str(depth))

        df = pd.DataFrame(columns=columns, data=self.paths, dtype=int)
        df.to_pickle(self.processed_dir + 'paths_' + str(depth) + '.pkl')

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
        self.name = 'FB15K-237'


class WN18RR(Dataset):
    def __init__(self):
        super().__init__('./data/WN18RR/raw/', './data/WN18RR/processed/', './data/WN18RR/evaluation/',
                         './data/WN18RR/run/')
        self.n = 40943
        self.k = 11
        self.size_x = 50
        self.size_g = 50
        self.name = 'WN18RR'
