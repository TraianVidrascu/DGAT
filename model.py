import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, h, g, edge_idx, edge_type):
        # self edges
        n = h.shape[0]
        self._self_edges_mask(n)

        # compute edge embeddings
        c_ijk = self._compute_edges(edge_idx, edge_type, h, g)

        # compute edge attention
        alpha = self._attention(c_ijk, edge_idx)

        # aggregate node representation
        h = self._aggregate(edge_idx, alpha, c_ijk)
        return h


class EntityLayer(nn.Module):
    def __init__(self, initial_size, heads, layer_size, dev='cpu'):
        super(EntityLayer, self).__init__()
        # entity embedding
        self.weights_ent = nn.Linear(initial_size, layer_size)
        self.init_params()
        self.to(dev)
        self.heads = heads

    def init_params(self):
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)

    def forward(self, x, h):
        h_prime = self.weights_ent(x)[:, None, :] + h

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


class DKBATNet(nn.Module):
    def __init__(self, x_size, g_size, hidden_size, output_size, heads, alpha=0.5, margin=1, device='cpu'):
        super(DKBATNet, self).__init__()
        self.inbound_input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, device=device)
        self.outbound_input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, device=device)

        self.inbound_output_layer = RelationalAttentionLayer(hidden_size * heads, g_size, output_size, heads,
                                                             device=device)
        self.outbound_output_layer = RelationalAttentionLayer(hidden_size * heads, g_size, output_size, heads,
                                                              device=device)

        self.entity_layer = EntityLayer(x_size, heads, hidden_size, device)
        self.relation_layer = RelationLayer(g_size, heads * hidden_size, device)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.device = device
        self.to(device)

        self.actv = nn.LeakyReLU()

    def _dissimilarity(self, h, g, edge_idx, edge_type):
        row, col = edge_idx
        d_norm = torch.norm(h[row] + g[edge_type] - h[col], p=1, dim=1)
        return d_norm

    def _concat(self, h):
        h = torch.cat([h[:, i, :] for i in range(self.heads)], dim=1)
        return h

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

    def _merge_heads(self, h):
        self.h = torch.sum(h, dim=1) / self.heads
        return h

    def forward(self, x, g, edge_idx, edge_type):
        x = F.normalize(x, p=2, dim=1).detach()
        torch.cuda.empty_cache()

        row, col = edge_idx
        outbound_edge_idx = torch.stack([col, row])

        h_inbound = self.inbound_input_layer(x, g, edge_idx, edge_type)
        h_outbound = self.outbound_input_layer(x, g, outbound_edge_idx, edge_type)
        h = self.alpha * h_inbound + (1 - self.alpha) * h_outbound
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)
        h = self._concat(h)

        h_inbound = self.inbound_output_layer(h, g, edge_idx, edge_type)
        h_outbound = self.outbound_output_layer(h, g, outbound_edge_idx, edge_type)
        h = self.alpha * h_inbound + (1 - self.alpha) * h_outbound
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)

        h_prime = self.entity_layer(x, h).view(-1, self.heads * self.output_size)
        h_prime = F.normalize(h_prime, p=2, dim=2)

        h_prime = self._merge_heads(h_prime)
        g_prime = self.relation_layer(g)

        return h_prime, g_prime


class KBNet(nn.Module):
    def __init__(self, x_size, g_size, hidden_size, output_size, heads, margin=1, device='cpu'):
        super(KBNet, self).__init__()
        self.input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, device=device)
        self.input_entity_layer = EntityLayer(x_size, heads, hidden_size, device)
        self.input_relation_layer = RelationLayer(g_size, heads * hidden_size, device)

        self.output_layer = RelationalAttentionLayer(heads * hidden_size, g_size, output_size, heads,
                                                     device=device)

        self.entity_layer = EntityLayer(x_size, heads, output_size, device)
        self.relation_layer = RelationLayer(g_size, output_size, device)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.device = device
        self.to(device)

        self.actv = nn.LeakyReLU()

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

    def _merge_heads(self, h):
        h = torch.sum(h, dim=1) / self.heads
        return h

    def _concat(self, h):
        h = torch.cat([h[:, i, :] for i in range(self.heads)], dim=1)
        return h

    def forward(self, x, g, edge_idx, edge_type):
        x = F.normalize(x, p=2, dim=1).detach()

        torch.cuda.empty_cache()

        h = self.input_layer(x, g, edge_idx, edge_type)
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)
        h = self._concat(h)

        h = self.output_layer(h, g, edge_idx, edge_type)
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)

        h_prime = self.entity_layer(x, h)
        g_prime = self.relation_layer(g)

        h_prime = F.normalize(h_prime, p=2, dim=2)
        h_prime = self._merge_heads(h_prime)

        return h_prime, g_prime


# class ConvKB(nn.Module):
#     def __init__(self, input_size, channels, dropout=0.3, dev='cpu'):
#         super(ConvKB, self).__init__()
#
#         self.conv = nn.Conv2d(1, channels, kernel_size=(3, 1), bias=True)
#         self.weight = nn.Linear(input_size * channels, 1)
#         self.dropout = nn.Dropout(dropout)
#
#         self.device = dev
#         self.channels = channels
#         self.input_size = input_size
#
#         self.args = input_size, channels, dropout, dev
#
#         self.loss_fct = nn.SoftMarginLoss()
#         self.to(dev)
#
#     def loss(self, y, t):
#         return self.loss_fct(y, t)
#
#     def evaluate(self, h, g, triplets_tail, triplets_tail_type):
#         with torch.no_grad():
#             self.eval()
#             scores = torch.detach(self(h, g, triplets_tail, triplets_tail_type).cpu()).numpy()
#         return scores
#
#     def forward(self, h, g, edge_idx, edge_type):
#         row, col = edge_idx
#
#         h = torch.cat([h[row][:, None, :], h[col][:, None, :], g[edge_type][:, None, :]], dim=1)[:, None, :, :]
#
#         h = self.conv(h).squeeze()
#         h = torch.relu(h)
#
#         h = h.view(-1, self.channels * self.input_size)
#         h = self.dropout(h)
#         h = self.weight(h).squeeze()
#
#         return h

class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob, dev='cpu'):
        super().__init__()

        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))  # kernel size -> 1*input_seq_length(i.e. 2)
        self.dropout = nn.Dropout(drop_prob)
        self.non_linearity = nn.ReLU()
        self.fc_layer = nn.Linear((input_dim) * out_channels, 1)

        nn.init.xavier_uniform_(self.fc_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)
        self.dev = dev
        self.to(dev)

    def forward(self, conv_input):
        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc_layer(input_fc)
        return output

    def evaluate(self, h, g, edge_idx, batch_type):
        with torch.no_grad():
            self.eval()
            row, col = edge_idx
            conv_input = torch.stack([h[row], g[batch_type], h[col]], dim=1).to(self.dev)

            scores = torch.detach(self(conv_input).view(-1).cpu()).numpy()
        return scores
