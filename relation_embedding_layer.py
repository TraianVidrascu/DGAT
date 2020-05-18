import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean


class RelationLayer(nn.Module):
    def __init__(self, in_size, out_size, device):
        super(RelationLayer, self).__init__()
        # relation layer
        self.weights_rel = nn.Linear(in_size, out_size, bias=True)
        self.weights_g = nn.Linear(in_size * 3, out_size, bias=False)
        self.init_params()

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)
        nn.init.xavier_normal_(self.weights_g.weight, gain=1.414)
        nn.init.zeros_(self.weights_rel.bias)

    def forward(self, g_initial, c_ijk, edge_type):
        g_w = self.weights_g(c_ijk)
        g_w = scatter_mean(g_w, edge_type, dim=0).squeeze()
        g_prime = self.weights_rel(g_initial) + g_w
        return g_prime


class SimpleRelationLayer(nn.Module):
    def __init__(self, in_size, out_size, device):
        super(SimpleRelationLayer, self).__init__()
        # relation layer
        self.weights_rel = nn.Linear(in_size, out_size, bias=False)
        self.init_params()

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)

    def forward(self, g):
        g_prime = self.weights_rel(g)
        return g_prime
