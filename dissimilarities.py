from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import torch.nn.functional as F


class Dissimilarity(nn.Module, ABC):

    @abstractmethod
    def forward(self, h, g, edge_idx, edge_type):
        pass


class TransE(Dissimilarity):
    def __init__(self):
        super(TransE, self).__init__()

    def forward(self, h, g, edge_idx, edge_type):
        torch.cuda.empty_cache()
        row, col = edge_idx
        d_norm = torch.norm(h[row, :] + g[edge_type] - h[col, :], p=1, dim=1)
        return d_norm


class TransR(Dissimilarity):
    def __init__(self, embedding_size, bias=True, dev='cpu'):
        super(TransR, self).__init__()
        self.projection = nn.Linear(embedding_size, embedding_size, bias=bias)
        self.to(dev)

    def forward(self, h, g, edge_idx, edge_type):
        torch.cuda.empty_cache()
        row, col = edge_idx
        s = self.projection(h[row, :])
        t = self.projection(h[col, :])
        d_norm = torch.norm(s + g[edge_type] - t, p=1, dim=1)
        return d_norm


class ModE(Dissimilarity):
    def __init__(self, dev='cpu'):
        super(ModE, self).__init__()
        self.dev = dev
        self.to(dev)

    def forward(self, h, g, edge_idx, edge_type):
        row, col = edge_idx
        d_norm = torch.norm(h[row, :] * g[edge_type, :] - h[col, :], p=2, dim=-1)
        return d_norm


class Hake(Dissimilarity):
    def __init__(self, embedding_size, dev='cpu'):
        super(Hake, self).__init__()
        self.dev = dev

        self.bias_estimator = nn.Linear(embedding_size, embedding_size, bias=True)
        self.phase_estimator = nn.Linear(embedding_size, embedding_size, bias=True)
        self.lambda_param = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.to(dev)
        self.init_params()

    def init_params(self):
        nn.init.xavier_normal_(self.bias_estimator.weight, gain=1.414)
        nn.init.xavier_normal_(self.phase_estimator.weight, gain=1.414)

        nn.init.zeros_(self.bias_estimator.bias)
        nn.init.zeros_(self.phase_estimator.bias)

    def forward(self, h, g, edge_idx, edge_type):
        row, col = edge_idx

        head_mod = h[row, :]
        tail_mod = h[col, :]
        rel_mod = torch.abs(g[edge_type, :])

        bias_mod = self.bias_estimator(rel_mod)
        bias_mod = torch.clamp(bias_mod, max=1)
        is_smaller = bias_mod < -rel_mod
        bias_mod[is_smaller] = rel_mod[is_smaller]

        mod_score = torch.norm(head_mod * rel_mod + (head_mod + tail_mod) * bias_mod - tail_mod, p=2, dim=-1)

        head_phase = 2 * torch.acos(self.phase_estimator(head_mod))
        tail_phase = 2 * torch.acos(self.phase_estimator(tail_mod))
        rel_phase = 2 * torch.acos(self.phase_estimator(rel_mod))

        phase_score = torch.norm(torch.sin((head_phase + rel_phase - tail_phase) / 2), p=1, dim=-1)

        score = mod_score + self.lambda_param * phase_score
        return score
