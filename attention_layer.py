import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add

from relation_embedding_layer import RelationLayer


class RelationalAttentionLayer(nn.Module):
    def __init__(self, in_size_h, in_size_g, out_size, heads=2, concat=True,
                 negative_slope=2e-1, dropout=0.3, device='cpu'):
        super(RelationalAttentionLayer, self).__init__()
        # forward layers
        self.fc1 = nn.Linear(2 * in_size_h + in_size_g, heads * out_size, bias=False)

        # attention layers
        self.weights_att = nn.Parameter(torch.Tensor(1, heads, out_size))
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.att_softmax = nn.Softmax()

        self.to(device)
        self.device = device

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # parameters
        self.concat = concat
        self.heads = heads
        self.in_size_h = in_size_h
        self.in_size_g = in_size_g
        self.out_size = out_size

        self.self_edges = None
        self.self_edge_idx = None

        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.weights_att.data, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)

    @staticmethod
    def _sofmax(indexes, egde_values):
        """

        :param indexes: nodes of each edge
        :param egde_values: values of each edge
        :return: normalized values of edges considering nodes
        """
        edge_values = torch.exp(egde_values)

        row_sum = scatter_add(edge_values, indexes, dim=0)

        edge_softmax = edge_values / row_sum[indexes, :, :]

        return edge_softmax

    def _concat(self, h):
        h = torch.cat([h[:, i, :] for i in range(self.heads)], dim=1)
        return h

    def _aggregate(self, end_nodes, alpha, c_ijk):
        a = alpha * c_ijk

        # add self edges
        a = torch.cat([a, self.self_edges.type_as(a)], dim=0)
        end_nodes = torch.cat([end_nodes, self.self_edge_mask.type_as(end_nodes)])

        h = scatter_add(a, end_nodes, dim=0)

        return h

    def _attention(self, c_ijk, end_nodes):
        a_ijk = torch.sum(self.weights_att * c_ijk, dim=2)[:, :, None]
        b_ijk = -self.leaky_relu(a_ijk)

        alpha = RelationalAttentionLayer._sofmax(end_nodes, b_ijk)
        alpha = self.dropout(alpha)
        return alpha

    def _compute_edges(self, h_ijk):
        # obtain edge representation
        c_ijk = self.fc1(h_ijk).view(-1, self.heads, self.out_size)

        return c_ijk

    def _self_edges_mask(self, n):
        """
        Adds self edges for with 0 values to keep the number nodes consistent per compuation in the case
        of an unconnected node
        :param n:
        """
        if self.self_edges is None:
            self.self_edges = torch.zeros((n, self.heads, self.out_size))
            self.self_edge_mask = torch.tensor([i for i in range(n)]).long()

    def forward(self, h_ijk, ends, n):
        # self edges
        self._self_edges_mask(n)

        # compute edge embeddings
        c_ijk = self._compute_edges(h_ijk)

        # compute edge attention
        alpha = self._attention(c_ijk, ends)

        # aggregate node representation
        h = self._aggregate(ends, alpha, c_ijk)
        h = F.elu(h)

        h = F.normalize(h, dim=2, p=2)
        if self.concat:
            h = self._concat(h)
        return h