# from stellargraph.data import EdgeSplitter
import torch

PROBS = [0.0, 0.25, 0.50, 0.25]
GLOBAL = 'global'


class DataLoader:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_properties(self):
        n = self.dataset.n
        k = self.dataset.k
        return n, k

    def get_name(self):
        return self.dataset.name

    def load(self, fold, dev='cpu'):
        x, g, graph = self.dataset.load_fold(fold, dev)
        return x, g, graph

    def load_train(self, dev='cpu'):
        x, g, graph = self.dataset.load_fold('train', dev)
        return x, g, graph

    def load_valid(self, dev='cpu'):
        x, g, graph = self.dataset.load_fold('valid', dev)
        return x, g, graph

    def load_test(self, dev='cpu'):
        x, g, graph = self.dataset.load_fold('test', dev)
        return x, g, graph

    def load_embedding(self, dev='cpu'):
        h, g = self.dataset.load_embedding(dev)
        return h, g

    # def split_unseen(self, graph, dev='cpu'):
    #     splitter = EdgeSplitter(graph)
    #     graph_reduced, edge_idx_unseen, edge_lbl_unseen = splitter.train_test_split(p=0.1,
    #                                                                                 method=GLOBAL,
    #                                                                                 probs=PROBS)
    #     edge_idx_unseen = edge_idx_unseen[edge_lbl_unseen == 1]
    #
    #     edge_unseen_type = list(map(lambda x: graph[x[0]][x[1]][0]['label'], edge_idx_unseen))
    #     edge_unseen_type = torch.tensor(edge_unseen_type).long()
    #
    #     edge_idx_unseen = torch.from_numpy(edge_idx_unseen).t()
    #
    #     return graph_reduced, edge_idx_unseen.to(dev), edge_unseen_type.to(dev)

    @staticmethod
    def graph2idx(graph, dev='cpu'):
        edges = list(map(lambda x: [x[0], x[1], x[2]['label']], graph.edges(data=True)))

        edges = torch.tensor(edges).long()
        edge_idx = edges[:, 0:2].t()
        edge_type = edges[:, 2]

        return edge_idx.to(dev), edge_type.to(dev)

    @staticmethod
    def corrupt_triplet(n, triplet, head=True):
        # corrupt triplet, raw setting now, corrupt only head or tail
        x, y = triplet

        entities = torch.randperm(n)
        triplets = torch.zeros(size=(2, n)).long()
        if head:
            triplets[0, :] = entities
            triplets[1, :] = y
            position = (triplets[0, :] == x).nonzero().item()
        else:
            triplets[1, :] = entities
            triplets[0, :] = x
            position = (triplets[1, :] == y).nonzero().item()
        return triplets, position

    def shuffle_samples(self, edge_idx, edge_type):
        m = edge_idx.shape[1]
        samples = torch.stack([edge_idx[0, :], edge_idx[1, :], edge_type], dim=0)

        perm = torch.randperm(m)
        samples = samples[:, perm]

        edge_idx = samples[0:2, :]
        edge_type = samples[2, :]

        return edge_idx, edge_type

    def load_evaluation_triplets_raw(self, fold, head=True, dev='cpu'):
        triplets, lists = self.dataset.load_evaluation_triplets_raw(fold, head, dev)
        return triplets, lists

    def negative_samples(self, n, edge_idx, edge_type, negative_ratio, dev='cpu'):
        ratio = int(negative_ratio / 2)

        edge_idx_aux = edge_idx.repeat((1, ratio))
        row, col = edge_idx_aux
        m = row.shape[0]

        edge_type_aux = edge_type.repeat(ratio)

        # corrupt head triplet
        head_corrupted = torch.randint(size=(m,), high=n).to(dev)
        head_corrupted[row == head_corrupted] = (head_corrupted[row == head_corrupted] + 1) % n
        head_corrupted_idx_type = torch.stack([head_corrupted, col, edge_type_aux])

        # corrupt tail triplet
        tail_corrupted = torch.randint(size=(m,), high=n).to(dev)
        tail_corrupted[col == tail_corrupted] = (tail_corrupted[col == tail_corrupted] + 1) % n
        tail_corrupted_idx_type = torch.stack([row, tail_corrupted, edge_type_aux])

        # negative samples
        neg_idx_type = torch.cat([head_corrupted_idx_type, tail_corrupted_idx_type], dim=1)

        neg_idx = neg_idx_type[0:2, :]
        neg_type = neg_idx_type[2, :]

        return neg_idx, neg_type
