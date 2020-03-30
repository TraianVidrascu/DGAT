import torch
import torch.nn as nn

from torch_scatter import scatter_add


class RelationalAttentionLayer(nn.Module):
    def __init__(self, in_size_h, in_size_g, out_size, heads=2, concat=True, bias=True,
                 negative_slope=1e-2, dropout=0.3, device='cpu'):
        super(RelationalAttentionLayer, self).__init__()
        # forward layers
        self.fc1 = nn.Linear(2 * in_size_h + in_size_g, heads * out_size, bias=bias)

        # attention layers
        self.weights_att = nn.Parameter(torch.Tensor(1, heads, out_size))
        self.att_actv = nn.LeakyReLU(negative_slope)
        self.att_softmax = nn.Softmax()

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
        self.to(device)

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
        return h


class EntityLayer(nn.Module):
    def __init__(self, initial_size, layer_size, dev='device'):
        super(EntityLayer, self).__init__()
        # entity embedding
        self.weights_ent = nn.Linear(initial_size, layer_size)
        self.init_params()
        self.to(dev)

    def init_params(self):
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)

    def forward(self, x, h):
        h_prime = self.weights_ent(x)[:, None, :] + h

        # concat representations
        h_prime = torch.cat([h_prime[:, i, :] for i in range(self.heads)], dim=1)
        h_prime = torch.tanh(h_prime)

        return h_prime


class RelationLayer(nn.Module):
    def __init__(self, in_size, out_size, device='cpu'):
        super(RelationLayer, self).__init__()
        # relation layer
        self.weights_rel = nn.Linear(in_size, out_size)
        self.init_params()
        self.to(device)

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)

    def forward(self, g):
        g_prime = self.weights_rel(g)
        return g_prime


class KBNet(nn.Module):
    def __init__(self, x_size, g_size, hidden_size, output_size, heads, margin=1, device='cpu'):
        super(KBNet, self).__init__()
        self.input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, device=device)
        self.output_layer = RelationalAttentionLayer(heads * hidden_size, g_size, output_size, heads, device=device)

        self.relation_layer = RelationLayer(g_size, heads * output_size)
        self.entity_layer = EntityLayer(x_size, heads * output_size)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.device = device
        self.to(device)

    def _dissimilarity(self, h, g, edge_idx, edge_type):
        row, col = edge_idx
        d_norm = torch.norm(h[row] + g[edge_type] - h[col], p=1, dim=1)
        return d_norm

    def loss(self, h_prime, g_prime, pos_edge_idx, pos_edge_type, neg_edge_idx, neg_edge_type):
        # Margin loss
        d_pos = self._dissimilarity(h_prime, g_prime, pos_edge_idx, pos_edge_type)
        d_neg = self._dissimilarity(h_prime, g_prime, neg_edge_idx, neg_edge_type)
        y = torch.ones(d_pos.shape[0]).to(d_pos.device)
        loss = self.loss_fct(d_pos, d_neg, y)
        return loss

    def evaluate(self, h, g, triplets_tail, triplets_tail_type):
        with torch.no_grad():
            self.eval()
            scores = torch.detach(self._dissimilarity(h, g, triplets_tail, triplets_tail_type).cpu()).numpy()
        return scores

    def forward(self, x, g, edge_idx, edge_type):
        h = self.input_layer(x, x, g, edge_idx, edge_type)
        h = self.output_layer(x, h, g, edge_idx, edge_type)

        g_prime = self.relation_layer(g)
        h_prime = self.entity_layer(h).view(-1, self.heads * self.output_layer)

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

        self.args = input_size, channels, dropout, dev

        self.loss_fct = nn.SoftMarginLoss()
        self.to(dev)

    def loss(self, y, t):
        return self.loss_fct(y, t)

    def evaluate(self, h, g, triplets_tail, triplets_tail_type):
        with torch.no_grad():
            self.eval()
            scores = torch.detach(self(h, g, triplets_tail, triplets_tail_type).cpu()).numpy()
        return scores

    def forward(self, h, g, edge_idx, edge_type):
        row, col = edge_idx

        h = torch.cat([h[row][:, None, :], h[col][:, None, :], g[edge_type][:, None, :]], dim=1)[:, None, :, :]

        h = self.conv(h).squeeze()
        h = torch.relu(h)
        h = self.dropout(h)
        h = h.view(-1, self.channels * self.input_size)

        h = self.weight(h).squeeze()

        return h
