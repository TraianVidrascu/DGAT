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

    def graph2idx(self, graph, dev='cpu'):
        edges = list(map(lambda x: [x[0], x[1], x[2]['label']], graph.edges(data=True)))

        edges = torch.tensor(edges).long()
        edge_idx = edges[:, 0:2].t()
        edge_type = edges[:, 2]

        return edge_idx.to(dev), edge_type.to(dev)

    def corrupt_triplet(self, n, triplet, head=True):
        # corrupt triplet, raw setting now, corrupt only head or tail
        x, y = triplet

        entities = torch.randperm(n)
        triplets = torch.zeros(size=(2,n)).long()
        if head:
            triplets[0,:] = entities
            triplets[1,:] = y
            position = (triplets[0, :] == x).nonzero().item()
        else:
            triplets[1, :] = entities
            triplets[0, :] = x
            position = (triplets[1, :] == y).nonzero().item()
        return triplets, position

    def negative_samples(self, n, edge_idx, dev='cpu'):
        row, col = edge_idx

        # corrupt head triplet
        head_corrupted = torch.randint_like(row, high=n)
        head_corrupted[row == head_corrupted] = (head_corrupted[row == head_corrupted] + 1) % n

        # corrupt tail triplet
        tail_corrupted = torch.randint_like(col, high=n)
        tail_corrupted[col == tail_corrupted] = (tail_corrupted[col == tail_corrupted] + 1) % n

        # uniform sample between two sets
        m = edge_idx.shape[1]
        sampled = torch.rand(m) > 0.5

        neg_idx = torch.zeros((2, m)).long().to(dev)
        neg_idx[0, sampled] = head_corrupted[sampled]
        neg_idx[0, ~sampled] = edge_idx[0, ~sampled]
        neg_idx[1, ~sampled] = tail_corrupted[~sampled]
        neg_idx[1, sampled] = edge_idx[1, sampled]

        return neg_idx
