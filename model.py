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

    def _aggregate(self, end_nodes, alpha, c_ijk):
        a = alpha * c_ijk

        # add self edges
        a = torch.cat([a, self.self_edges.type_as(a)], dim=0)
        end_nodes = torch.cat([end_nodes, self.self_edge_mask.type_as(end_nodes)])

        h = scatter_add(a, end_nodes, dim=0)
        return h

    def _attention(self, c_ijk, end_nodes):
        a_ijk = torch.sum(self.weights_att * c_ijk, dim=2)[:, :, None]
        b_ijk = self.att_actv(a_ijk)

        alpha = RelationalAttentionLayer._sofmax(end_nodes, b_ijk)

        alpha = self.dropout(alpha)
        return alpha

    def _compute_edges(self, edge_idx, edge_type, h, g):
        row, col = edge_idx

        # extract embeddings for entities and relations
        h_i = h[row]
        h_j = h[col]

        # add zero embeddings for paths of only one hop
        g_size = g.shape[1]
        g_zeros = torch.zeros((1, g_size)).type_as(g).to(g.device)
        g_aux = torch.cat([g, g_zeros], dim=0)

        k_1, k2 = edge_type
        g_k = g_aux[k_1] + g_aux[k2]

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
            self.self_edges = torch.zeros((n, self.heads, self.out_size))
            self.self_edge_mask = torch.tensor([i for i in range(n)]).long()

    def forward(self, h, g, edge_idx, edge_type):
        # self edges
        n = h.shape[0]
        self._self_edges_mask(n)
        rows = edge_idx[0, :]
        cols = edge_idx[1, :]

        # compute edge embeddings
        c_ijk = self._compute_edges(edge_idx, edge_type, h, g)

        # compute edge attention
        alpha = self._attention(c_ijk, cols)

        # aggregate node representation
        h = self._aggregate(cols, alpha, c_ijk)
        return h


class EntityLayer(nn.Module):
    def __init__(self, initial_size, heads, layer_size, device='cpu'):
        super(EntityLayer, self).__init__()
        # entity embedding
        self.weights_ent = nn.Linear(initial_size, layer_size)
        self.init_params()
        self.to(device)
        self.heads = heads

    def init_params(self):
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)

    def forward(self, x, h):
        h_prime = self.weights_ent(x)[:, None, :] + h

        return h_prime


class RelationLayer(nn.Module):
    def __init__(self, in_size, out_size, device):
        super(RelationLayer, self).__init__()
        # relation layer
        self.weights_rel = nn.Linear(in_size, out_size)
        self.init_params()

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)

    def forward(self, g):
        g_prime = self.weights_rel(g)
        return g_prime


class AlphaLayer(nn.Module):
    def __init__(self, h_size, device='cpu'):
        super(AlphaLayer, self).__init__()
        self.alpha = nn.Linear(2 * h_size, 1)
        self.actv = nn.Sigmoid()

        self.init_params()

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.alpha.weight, gain=1.414)

    def forward(self, h_inbound, h_outbound):
        h_all = torch.cat([h_inbound, h_outbound], dim=2)
        alpha = self.alpha(h_all)
        alpha = self.actv(alpha)
        return alpha


class KB(nn.Module):
    def __init__(self):
        super(KB, self).__init__()

    @staticmethod
    def _dissimilarity(h, g, edge_idx, edge_type):
        row, col = edge_idx

        if len(edge_type.shape) == 2:
            g_size = g.shape[1]
            g_zeros = torch.zeros((1, g_size)).type_as(g).to(g.device)
            g_aux = torch.cat([g, g_zeros], dim=0)
            k_1, k2 = edge_type
            d_norm = torch.norm(h[row] + g_aux[k_1] + g_aux[k2] - h[col], p=1, dim=1)
        else:
            d_norm = torch.norm(h[row] + g[edge_type] - h[col], p=1, dim=1)
        return d_norm

    def loss(self, h_prime, g_prime, pos_edge_idx, pos_edge_type, neg_edge_idx, neg_edge_type):
        # Margin loss
        d_pos = KB._dissimilarity(h_prime, g_prime, pos_edge_idx, pos_edge_type)
        d_neg = KB._dissimilarity(h_prime, g_prime, neg_edge_idx, neg_edge_type)
        y = torch.ones(d_pos.shape[0]).to(d_pos.device)
        loss = self.loss_fct(d_pos, d_neg, y)
        return loss

    def evaluate(self, h, g, triplets, triplets_type):
        with torch.no_grad():
            self.eval()
            scores = torch.detach(KB._dissimilarity(h, g, triplets, triplets_type).cpu()).numpy()
        return scores

    def _merge_heads(self, h):
        h = torch.sum(h, dim=1) / self.heads
        return h

    def _concat(self, h):
        h = torch.cat([h[:, i, :] for i in range(self.heads)], dim=1)
        return h


class DKBATNet(KB):
    def __init__(self, x_size, g_size, hidden_size, output_size, heads, alpha=0.5, margin=1, dropout=0.3,
                 negative_slope=0.2,
                 device='cpu'):
        super(DKBATNet, self).__init__()
        self.inbound_input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, dropout=dropout,
                                                            device='cuda:1')
        self.outbound_input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, dropout=dropout,
                                                             device='cuda:2')
        self.alpha_input = AlphaLayer(hidden_size)

        self.inbound_output_layer = RelationalAttentionLayer(hidden_size * heads, g_size, output_size, heads,
                                                             dropout=dropout,
                                                             device='cuda:1')
        self.outbound_output_layer = RelationalAttentionLayer(hidden_size * heads, g_size, output_size, heads,
                                                              dropout=dropout,
                                                              device='cuda:2')

        self.alpha_output = AlphaLayer(output_size, 'cuda:0')

        self.entity_layer = EntityLayer(x_size, heads, output_size, device='cuda:1')
        self.relation_layer = RelationLayer(g_size, output_size, device='cuda:2')

        self.loss_fct = nn.MarginRankingLoss(margin=margin).to('cuda:0')

        self.heads = heads
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.alpha = alpha

        self.actv = nn.LeakyReLU(negative_slope)

    def forward(self, x, g, edge_idx, edge_type):
        x = F.normalize(x, p=2, dim=1).detach()
        torch.cuda.empty_cache()

        row, col = edge_idx
        outbound_edge_idx = torch.stack([col, row])

        h_inbound = self.inbound_input_layer(x.to('cuda:1'), g.to('cuda:1'), edge_idx.to('cuda:1'),
                                             edge_type.to('cuda:1'))
        h_outbound = self.outbound_input_layer(x.to('cuda:2'), g.to('cuda:2'), outbound_edge_idx.to('cuda:2'),
                                               edge_type.to('cuda:2'))
        h_inbound, h_outbound = h_inbound.to('cuda:0'), h_outbound.to('cuda:0')

        alpha = self.alpha_input(h_inbound, h_outbound)
        h = alpha * h_inbound + (1 - alpha) * h_outbound
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)
        h = self._concat(h)

        torch.cuda.empty_cache()

        h_inbound = self.inbound_output_layer(h.to('cuda:1'), g.to('cuda:1'), edge_idx.to('cuda:1'),
                                              edge_type.to('cuda:1'))
        h_outbound = self.outbound_output_layer(h.to('cuda:2'), g.to('cuda:2'), outbound_edge_idx.to('cuda:2'),
                                                edge_type.to('cuda:2'))
        h_inbound, h_outbound = h_inbound.to('cuda:0'), h_outbound.to('cuda:0')

        alpha = self.alpha_output(h_inbound, h_outbound)
        h = alpha * h_inbound + (1 - alpha) * h_outbound
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)

        h_prime = self.entity_layer(x.to('cuda:1'), h.to('cuda:1'))
        h_prime = F.normalize(h_prime, p=2, dim=2)

        h_prime = self._merge_heads(h_prime)
        g_prime = self.relation_layer(g.to('cuda:2'))

        return h_prime.to('cuda:0'), g_prime.to('cuda:0')


class KBNet(KB):
    def __init__(self, x_size, g_size, hidden_size, output_size, heads, margin=1, dropout=0.3, negative_slope=0.2,
                 device='cpu'):
        super(KBNet, self).__init__()
        self.input_layer = RelationalAttentionLayer(x_size, g_size, hidden_size, heads, dropout=dropout,
                                                    device='cuda:1')
        self.output_layer = RelationalAttentionLayer(heads * hidden_size, g_size, output_size, heads, dropout=dropout,
                                                     device='cuda:1')

        self.entity_layer = EntityLayer(x_size, heads, output_size, device='cuda:2')
        self.relation_layer = RelationLayer(g_size, output_size, device='cuda:2')

        self.loss_fct = nn.MarginRankingLoss(margin=margin).to('cuda:0')

        self.heads = heads
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.actv = nn.LeakyReLU(negative_slope)

    def forward(self, x, g, edge_idx, edge_type):
        x = F.normalize(x, p=2, dim=1).detach()

        torch.cuda.empty_cache()

        h = self.input_layer(x.to('cuda:1'), g.to('cuda:1'), edge_idx.to('cuda:1'), edge_type.to('cuda:1'))
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)
        h = self._concat(h)

        h = self.output_layer(h.to('cuda:1'), g.to('cuda:1'), edge_idx.to('cuda:1'), edge_type.to('cuda:1'))
        h = self.actv(h)
        h = F.normalize(h, p=2, dim=2)

        h_prime = self.entity_layer(x.to('cuda:2'), h.to('cuda:2'))
        g_prime = self.relation_layer(g.to('cuda:2'))

        h_prime = F.normalize(h_prime, p=2, dim=2)
        h_prime = self._merge_heads(h_prime)

        return h_prime.to('cuda:0'), g_prime.to('cuda:0')


class ConvKB(nn.Module):
    def __init__(self, input_dim, input_seq_len, in_channels, out_channels, drop_prob=0.0, dev='cpu'):
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
