from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dissimilarity(nn.Module, ABC):

    @abstractmethod
    def get_score(self, h, g, edge_idx, edge_type):
        pass


class TransE(Dissimilarity):
    def __init__(self):
        super(TransE, self).__init__()

    def get_score(self, h, g, edge_idx, edge_type):
        torch.cuda.empty_cache()
        row, col = edge_idx
        d_norm = torch.norm(h[row, :] + g[edge_type] - h[col, :], p=1, dim=1)
        return d_norm


class TransR(Dissimilarity):
    def __init__(self, embedding_size, bias=True, dev='cpu'):
        super(TransR, self).__init__()
        self.projection = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.to(dev)

    def get_score(self, h, g, edge_idx, edge_type):
        torch.cuda.empty_cache()
        row, col = edge_idx
        s = self.projection(h[row, :])
        t = self.projection(h[col, :])
        d_norm = torch.norm(s + g[edge_type] - t, p=1, dim=1)
        return d_norm
