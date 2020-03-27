import torch
import torch.nn as nn

from torch_scatter import scatter_add


class RelationalAttentionLayer(nn.Module):
    def __init__(self, initial_size, in_size_h, in_size_g, out_size, heads=2, concat=True, bias=True,
                 negative_slope=1e-2, dropout=0.3, device='cpu'):
        super(RelationalAttentionLayer, self).__init__()
        # forward layers
        self.fc1 = nn.Linear(2 * in_size_h + in_size_g, heads * out_size, bias=bias)

        # attention layers
        self.weights_att = nn.Parameter(torch.Tensor(1, heads, out_size))
        self.att_actv = nn.LeakyReLU(negative_slope)
        self.att_softmax = nn.Softmax()

        # relation layer
        self.weights_rel = nn.Linear(in_size_g, out_size)
        # entity embedding
        self.weights_ent = nn.Linear(initial_size, out_size)

        # dropout layer
        self.dropout = nn.Dropout(dropout)

        # parameters
        self.concat = concat
        self.heads = heads
        self.in_size_h = in_size_h
        self.in_size_g = in_size_g
        self.out_size = out_size
        self.device = device

        self.self_edges = None
        self.self_edge_idx = None

        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.weights_att.data, gain=1.414)
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)
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

    def _aggregate(self, edge_idx, alpha, c_ijk):
        cols = edge_idx[1, :]
        a = alpha * c_ijk

        # add self edges
        a = torch.cat([a, self.self_edges], dim=0)
        cols = torch.cat([cols, self.self_edge_mask])

        h = scatter_add(a, cols, dim=0)
        return h

    def _attention(self, c_ijk, edge_idx):
        a_ijk = torch.sum(self.weights_att * c_ijk, dim=2)[:, :, None]
        b_ijk = self.att_actv(a_ijk)

        rows = edge_idx[0, :]
        alpha = RelationalAttentionLayer._sofmax(rows, b_ijk)

        alpha = self.dropout(alpha)
        return alpha

    def _compute_edges(self, edge_idx, edge_type, h, g):
        row, col = edge_idx

        # extract embeddings for entities and relations
        h_i = h[row]
        h_j = h[col]
        g_k = g[edge_type]

        # concatenate the 3 representations
        h_ijk = torch.cat([h_i, h_j, g_k], dim=1)
        # obtain edge representation
        c_ijk = self.fc1(h_ijk).view(-1, self.heads, self.out_size)

        return c_ijk

    def _update_relation(self, g):
        g_prime = self.weights_rel(g)
        return g_prime

    def _update_entity(self, x, h):
        h_prime = self.weights_ent(x)[:, None, :] + h
        return h_prime

    def _concat(self, h_prime):
        if self.concat:
            # ToDo: Check if view reorders correctly
            h_prime = torch.cat([h_prime[:, i, :] for i in range(self.heads)], dim=1)
            h_prime = torch.tanh(h_prime)
        else:
            h_prime /= self.heads
            h_prime = torch.sum(h_prime, dim=1).squeeze()
        return h_prime

    def _self_edges_mask(self, n):
        """
        Adds self edges for with 0 values to keep the number nodes consistent per compuation in the case
        of an unconnected node
        :param n:
        """
        if self.self_edges is None:
            self.self_edges = torch.zeros((n, self.heads, self.out_size)).to(self.device)
            self.self_edge_mask = torch.tensor([i for i in range(n)]).long().to(self.device)

    def forward(self, x, h, g, edge_idx, edge_type):
        # self edges
        n = x.shape[0]
        self._self_edges_mask(n)

        # compute edge embeddings
        c_ijk = self._compute_edges(edge_idx, edge_type, h, g)

        # compute edge attention
        alpha = self._attention(c_ijk, edge_idx)

        # aggregate node representation
        h = self._aggregate(edge_idx, alpha, c_ijk)

        h_prime = self._update_entity(x, h)

        h_prime = self._concat(h_prime)

        g_prime = self._update_relation(g)
        return h_prime, g_prime


class KBNet(nn.Module):
    def __init__(self, in_size_h, in_size_g, hidden_size, output_size, heads, margin=1, device='cpu'):
        super(KBNet, self).__init__()
        self.input_layer = RelationalAttentionLayer(in_size_h, in_size_h, in_size_g, hidden_size, heads, device=device)
        self.output_layer = RelationalAttentionLayer(in_size_h, heads * hidden_size, hidden_size, output_size,
                                                     heads=heads, concat=False, device=device)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.device = device
        self.to(device)

    def _dissimilarity(self, h, g, edge_idx, edge_type):
        row, col = edge_idx

        h_i = h[row]
        h_j = h[col]
        g_k = g[edge_type]

        d = h_i + g_k - h_j

        d_norm = torch.norm(d, p=1, dim=1)
        return d_norm

    def loss(self, h_prime, g_prime, pos_edge_idx, neg_edge_pos, edge_type):
        # Margin loss
        d_pos = self._dissimilarity(h_prime, g_prime, pos_edge_idx, edge_type)
        d_neg = self._dissimilarity(h_prime, g_prime, neg_edge_pos, edge_type)
        y = torch.ones(d_pos.shape[0]).to(d_pos.device)
        loss = self.loss_fct(d_pos, d_neg, y)
        return loss

    def forward(self, x, g, edge_idx, edge_type):
        h_prime, g_prime = self.input_layer(x, x, g, edge_idx, edge_type)
        h_prime, g_prime = self.output_layer(x, h_prime, g_prime, edge_idx, edge_type)

        return h_prime, g_prime


class ConvKB(nn.Module):
    def __init__(self, input_size, channels, dropout=0.3, dev='cpu'):
        super(ConvKB, self).__init__()

        self.conv = nn.Conv2d(1, channels, kernel_size=(3, 1), bias=True)
        self.weight = nn.Linear(input_size * channels, 1)
        self.dropout = nn.Dropout(dropout)

        self.device = dev
        self.channels = channels
        self.input_size = input_size

        self.loss_fct = nn.SoftMarginLoss()
        self.to(dev)

    def loss(self, y, t):
        return self.loss_fct(y, t)

    def forward(self, h, g, edge_idx, edge_type):
        row, col = edge_idx

        h = torch.cat([h[row][:, None, :], h[col][:, None, :], g[edge_type][:, None, :]], dim=1)[:, None, :, :]

        h = self.conv(h).squeeze()
        h = torch.relu(h)
        h = self.dropout(h)
        h = h.view(-1, self.channels * self.input_size)

        h = self.weight(h).squeeze()

        return h
