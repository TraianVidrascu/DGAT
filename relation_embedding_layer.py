import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


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
        g = scatter_add(c_ijk, edge_type, dim=0)
        g_prime = self.weights_rel(g_initial) + self.weights_g(g)
        g_prime = F.normalize(g_prime, p=2, dim=-1)
        return g_prime
