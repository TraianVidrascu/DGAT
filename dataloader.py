import torch
import ast

BEGINLIST = 'BEGIN'
ENDLIST = 'END'


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

    def load_embedding(self, model_name, dev='cpu'):
        h, g = self.dataset.load_embedding(model_name, dev)
        return h, g

    def load_paths(self):
        paths = self.dataset.load_paths()
        row = torch.tensor(paths['x'].values).long()
        col = torch.tensor(paths['y'].values).long()
        k_1 = torch.tensor(paths['g_1'].values).long()
        k_2 = torch.tensor(paths['g_2'].values).long()

        edge_idx = torch.stack([row, col])
        edge_type = torch.stack([k_1, k_2])
        return edge_idx, edge_type

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

    def get_filtered_eval_file(self, fold, head):
        file = self.dataset.get_filtered_eval_file(fold, head)
        return file

    @staticmethod
    def load_list(file, head, dev='cpu'):
        position = -1
        line = file.readline()
        if line == '':
            return None, None, position
        elif BEGINLIST in line:
            # read correct triplet
            line = file.readline()
            correct_triplet = torch.tensor(ast.literal_eval(line)).to(dev)

            # read corrupted triplets
            line = file.readline()
            array = ast.literal_eval(line)
            corrupted_triplets = torch.tensor(array).to(dev)

            correct_part = correct_triplet[0]
            correct_type = correct_triplet[1]
            other_part = correct_triplet[2]

            if head:
                edge_idx = torch.stack([corrupted_triplets, correct_part.expand(corrupted_triplets.shape[0])])
                correct_triplet = torch.tensor([other_part, correct_part]).to(dev)
            else:
                edge_idx = torch.stack([correct_part.expand(corrupted_triplets.shape[0]), corrupted_triplets])
                correct_triplet = torch.tensor([correct_part, other_part]).to(dev)
            # append correct triplet to first position
            correct_triplet = correct_triplet[:, None]
            edge_idx = torch.cat([correct_triplet, edge_idx], dim=1)
            edge_type = correct_type.expand(corrupted_triplets.shape[0] + 1)
            position = 0

            file.readline()
            return edge_idx, edge_type, position
        return None, None, position

    @staticmethod
    def negative_samples(n, edge_idx, edge_type, negative_ratio, dev='cpu'):
        # put negative sample of same edge together
        pos_edge_idx = edge_idx.repeat((negative_ratio, 1)).t().flatten().view(-1, 2).t()
        edge_type = edge_type[None, :].repeat((negative_ratio, 1)).t().flatten()

        row, col = pos_edge_idx
        m = row.shape[0]

        # corrupt head triplet
        head_corrupted = torch.randint(size=(m,), high=n).to(dev)
        head_corrupted[row == head_corrupted] = (head_corrupted[row == head_corrupted] + 1) % n

        # corrupt tail triplet
        tail_corrupted = torch.randint(size=(m,), high=n).to(dev)
        tail_corrupted[col == tail_corrupted] = (tail_corrupted[col == tail_corrupted] + 1) % n

        # negative samples, bernoulli sample tail or head
        sample = (torch.rand(size=(m,)) > 0.5).long()
        neg_idx = torch.zeros_like(pos_edge_idx)
        neg_idx[0, sample == 0] = head_corrupted[sample == 0]
        neg_idx[0, sample != 0] = pos_edge_idx[0, sample != 0]
        neg_idx[1, sample == 1] = tail_corrupted[sample == 1]
        neg_idx[1, sample != 1] = pos_edge_idx[1, sample != 1]

        return pos_edge_idx.to(dev), neg_idx.to(dev), edge_type.to(dev)
