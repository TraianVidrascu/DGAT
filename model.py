import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add


class RelationalAttentionLayer(nn.Module):
    def __init__(self, in_size_h, in_size_g, out_size, heads=2, concat=True, bias=True,
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


class EntityLayer(nn.Module):
    def __init__(self, initial_size, layer_size, device='cpu'):
        super(EntityLayer, self).__init__()
        # entity embedding
        self.weights_ent = nn.Linear(initial_size, layer_size, bias=False)
        self.init_params()
        self.to(device)

    def init_params(self):
        nn.init.xavier_normal_(self.weights_ent.weight, gain=1.414)

    def forward(self, x, h):
        h_prime = self.weights_ent(x) + h

        return h_prime


class RelationLayer(nn.Module):
    def __init__(self, in_size, out_size, h_size, negative_slope, dropout, device):
        super(RelationLayer, self).__init__()
        # relation layer
        self.weights_rel = nn.Linear(in_size, out_size, bias=True)
        self.fc1 = nn.Linear(in_size + 2 * h_size, out_size, bias=True)
        self.init_params()

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        self.to(device)
        self.device = device

    def init_params(self):
        nn.init.xavier_normal_(self.weights_rel.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc1.weight, gain=1.414)
        nn.init.zeros_(self.weights_rel.bias)
        nn.init.zeros_(self.fc1.bias)

    def forward(self, h_ijk, g, edge_type):
        g_edges = scatter_add(h_ijk, edge_type, dim=0)
        g_edges = F.normalize(g_edges, dim=1, p=2)
        g_edges = F.elu(g_edges)

        g_prime = self.weights_rel(g) + self.fc1(g_edges)

        return g_prime


class MergeLayer(nn.Module):
    def __init__(self, h_size, device='cpu'):
        super(MergeLayer, self).__init__()
        self.weight_inbound = nn.Linear(h_size, h_size, bias=True)
        self.weight_outbound = nn.Linear(h_size, h_size, bias=True)
        self.lambda_layer = nn.Linear(h_size * 2, 1, bias=True)
        self.init_params()
        self.to(device)

    def forward(self, h_inbound, h_outbound):
        h_inbound = self.weight_inbound(h_inbound)
        h_outbound = self.weight_outbound(h_outbound)
        lambda_param = self.lambda_layer(torch.cat([h_inbound, h_outbound], dim=1))
        lambda_param = torch.sigmoid(lambda_param)
        h = lambda_param * h_inbound + (1 - lambda_param) * h_outbound
        h = F.elu(h)
        h = F.normalize(h, dim=1, p=2)
        return h

    def init_params(self):
        nn.init.xavier_normal_(self.weight_inbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.weight_outbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.lambda_layer.weight, gain=1.414)

        nn.init.zeros_(self.weight_inbound.bias)
        nn.init.zeros_(self.weight_outbound.bias)
        nn.init.zeros_(self.lambda_layer.bias)


class KB(nn.Module):
    def __init__(self, x, g, dev='cpu'):
        super(KB, self).__init__()
        self.x_size = x.shape[1]
        self.g_size = g.shape[1]
        self.n = x.shape[0]
        self.m = g.shape[0]

        self.x_initial = nn.Parameter(x, requires_grad=False)
        self.g_initial = nn.Parameter(g, requires_grad=False)
        self.to(dev)

    @staticmethod
    def _dissimilarity(h, g, edge_idx, edge_type):
        torch.cuda.empty_cache()
        row, col = edge_idx
        d_norm = torch.norm(h[row, :] + g[edge_type, :] - h[col, :], p=1, dim=1)
        return d_norm

    def loss(self, h_prime, g_prime, pos_edge_idx, pos_edge_type, neg_edge_idx, neg_edge_type):
        # Margin loss
        d_pos = KB._dissimilarity(h_prime, g_prime, pos_edge_idx, pos_edge_type)
        d_neg = KB._dissimilarity(h_prime, g_prime, neg_edge_idx, neg_edge_type)
        y = torch.ones(d_pos.shape[0]).to(d_pos.device)
        loss = self.loss_fct(d_pos, d_neg, y)
        return loss

    def evaluate(self, h, g, eval_idx, eval_type):
        with torch.no_grad():
            self.eval()
            scores = torch.detach(KB._dissimilarity(h, g, eval_idx, eval_type).cpu())
            return scores


class DKBATNet(KB):
    def __init__(self, x, g, output_size, heads, margin=1, dropout=0.3,
                 negative_slope=0.2,
                 device='cpu'):
        super(DKBATNet, self).__init__(x, g, device)
        self.inbound_input_layer = RelationalAttentionLayer(self.x_size, self.g_size, output_size, heads,
                                                            negative_slope=negative_slope,
                                                            dropout=dropout,
                                                            device=device)
        self.outbound_input_layer = RelationalAttentionLayer(self.x_size, self.g_size, output_size, heads,
                                                             negative_slope=negative_slope,
                                                             dropout=dropout,
                                                             device=device)
        self.merge_layer_input = MergeLayer(output_size * heads, device)

        self.inbound_output_layer = RelationalAttentionLayer(output_size * heads, output_size * heads,
                                                             output_size * heads, 1,
                                                             negative_slope=negative_slope,
                                                             dropout=dropout,
                                                             device=device)
        self.outbound_output_layer = RelationalAttentionLayer(output_size * heads, output_size * heads,
                                                              output_size * heads, 1,
                                                              negative_slope=negative_slope,
                                                              dropout=dropout,
                                                              device=device)

        self.merge_layer_output = MergeLayer(output_size * heads, device)

        self.entity_layer = EntityLayer(self.x_size, heads * output_size, device=device)
        self.relation_layer = RelationLayer(self.g_size, output_size * heads, self.x_size, negative_slope,
                                            dropout, device=device)

        self.dropout = nn.Dropout(dropout)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size

        self.to(device)

    def forward(self, edge_idx, edge_type, path_idx, path_type, use_path):
        torch.cuda.empty_cache()
        self.x_initial.data = F.normalize(
            self.x_initial.data, p=2, dim=1).detach()

        row, col = edge_idx
        rel = edge_type
        ends_col = col
        ends_row = row
        if use_path:
            row_path, col_path = path_idx
            rel_1, rel_2 = path_type
            ends_col = torch.cat([col, col_path])
            ends_row = torch.cat([row, row_path])

        # compute h_ijk
        h_ijk = torch.cat([self.x_initial[row, :], self.x_initial[col, :], self.g_initial[rel, :]], dim=1)

        # compute h_ijk path
        if use_path:
            h_ijk_path = torch.cat([self.x_initial[row_path, :], self.x_initial[col_path, :],
                                    self.g_initial[rel_1, :] + self.g_initial[rel_2, :]], dim=1)

            # merge direct edges and paths
            h_ijk = torch.cat([h_ijk, h_ijk_path], dim=0)

        h_inbound = self.inbound_input_layer(h_ijk, ends_col, self.n)
        h_outbound = self.outbound_input_layer(h_ijk, ends_row, self.n)

        h = self.merge_layer_input(h_inbound, h_outbound)
        h = self.dropout(h)

        # compute relation embedding
        g_prime = self.relation_layer(h_ijk, self.g_initial, edge_type)

        # compute edge representation for second layer
        h_ijk = torch.cat([h[row, :], h[col, :], g_prime[rel, :]], dim=1)

        # compute h_ijk path
        if use_path:
            h_ijk_path = torch.cat([h[row_path, :], h[col_path, :],
                                    g_prime[rel_1, :] + g_prime[rel_2, :]], dim=1)

            # merge direct edges and paths
            h_ijk = torch.cat([h_ijk, h_ijk_path], dim=0)

        h_inbound = self.inbound_output_layer(h_ijk, ends_col, self.n)
        h_outbound = self.outbound_output_layer(h_ijk, ends_row, self.n)

        h = self.merge_layer_output(h_inbound, h_outbound)

        h_prime = self.entity_layer(self.x_initial, h)

        h_prime = F.normalize(h_prime, p=2, dim=1)

        return h_prime, g_prime


class KBNet(KB):
    def __init__(self, x, g, output_size, heads, margin=1, dropout=0.3, negative_slope=0.2,
                 device='cpu'):
        super(KBNet, self).__init__(x, g, device)
        self.input_attention_layer = RelationalAttentionLayer(self.x_size, self.g_size, output_size, heads,
                                                              negative_slope=negative_slope,
                                                              dropout=dropout,
                                                              device=device)
        self.output_attention_layer = RelationalAttentionLayer(output_size * heads, output_size * heads,
                                                               output_size * heads, 1, dropout=dropout,
                                                               negative_slope=negative_slope,
                                                               device=device, concat=False)

        self.entity_layer = EntityLayer(self.x_size, heads * output_size, device=device)
        self.relation_layer = RelationLayer(self.g_size, output_size * heads, self.x_size, negative_slope, dropout,
                                            device=device)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size

        self.dropout = nn.Dropout(dropout)
        self.to(device)

    def forward(self, edge_idx, edge_type, path_idx, path_type, use_path):
        torch.cuda.empty_cache()
        self.x_initial.data = F.normalize(
            self.x_initial.data, p=2, dim=1).detach()

        row, col = edge_idx
        rel = edge_type
        ends = col
        if use_path:
            row_path, col_path = path_idx
            rel_1, rel_2 = path_type
            ends = torch.cat([col, col_path])

        # compute h_ijk
        h_ijk = torch.cat([self.x_initial[row, :], self.x_initial[col, :], self.g_initial[rel, :]], dim=1)

        # compute h_ijk path
        if use_path:
            h_ijk_path = torch.cat([self.x_initial[row_path, :], self.x_initial[col_path, :],
                                    self.g_initial[rel_1, :] + self.g_initial[rel_2, :]], dim=1)

            # merge direct edges and paths
            h_ijk = torch.cat([h_ijk, h_ijk_path], dim=0)

        # compute embedding
        h = self.input_attention_layer(h_ijk, ends, self.n)

        h = self.dropout(h)

        g_prime = self.relation_layer(h_ijk, self.g_initial, edge_type)

        # computer edge representation for second layer
        h_ijk = torch.cat([h[row, :], h[col, :], g_prime[rel, :]], dim=1)

        # compute h_ijk path
        if use_path:
            h_ijk_path = torch.cat([h[row_path, :], h[col_path, :],
                                    g_prime[rel_1, :] + g_prime[rel_2, :]], dim=1)

            # merge direct edges and paths
            h_ijk = torch.cat([h_ijk, h_ijk_path], dim=0)

        h = self.output_attention_layer(h_ijk, ends, self.n).squeeze()

        # add initial embeddings to last layer
        h_prime = self.entity_layer(self.x_initial, h)

        # normalize last layer
        h_prime = F.normalize(h_prime, p=2, dim=1)

        return h_prime, g_prime


class WrapperConvKB(nn.Module):
    def __init__(self, h, g, input_dim, input_seq_len, in_channels, out_channels, drop_prob=0.0, dev='cpu'):
        super(WrapperConvKB, self).__init__()
        self.conv = ConvKB(input_dim, input_seq_len, in_channels, out_channels, drop_prob, dev)

        self.node_embeddings = nn.Parameter(h, requires_grad=True)
        self.rel_embeddings = nn.Parameter(g, requires_grad=True)

        self.dev = dev
        self.out_channels = out_channels
        self.to(dev)

    def forward(self, edge_idx, edge_type):
        row, col = edge_idx

        h_ijk = torch.stack(
            [self.node_embeddings[row, :], self.rel_embeddings[edge_type, :], self.node_embeddings[col, :]],
            dim=1).to(self.dev)

        preds = self.conv(h_ijk)
        return preds

    def evaluate(self, _, __, edge_idx, edge_type):
        with torch.no_grad():
            self.eval()
            n = edge_idx.shape[1]

            if n > 15000:
                step = n // 4
                scores = []
                for i in range(0, n, step):
                    batch_idx, batch_type = edge_idx[:, i:i + step], edge_type[i:i + step]
                    preds = torch.detach(self.forward(batch_idx, batch_type).view(-1).cpu())
                    scores.append(preds)
                scores = torch.cat(scores)
            else:
                scores = torch.detach(self.forward(edge_idx, edge_type).view(-1).cpu())
        return scores


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


class PartModel(nn.Module):
    def __init__(self, x_initial, g_initial, input_dim, input_seq_len, in_channels, out_channels=50, drop_prob=0.0,
                 dev='cpu'):
        super().__init__()
        self.conv_layer = nn.Conv2d(
            in_channels, out_channels, (1, input_seq_len))

        self.fc1_layer = nn.Linear((input_dim) * out_channels, 50)
        self.fc2_layer = nn.Linear(50, 1)

        self.x_initial = nn.Parameter(x_initial.clone(), requires_grad=False)
        self.g_initial = nn.Parameter(g_initial.clone(), requires_grad=False)

        self.x = nn.Parameter(x_initial, requires_grad=True)
        self.g = nn.Parameter(g_initial, requires_grad=True)

        nn.init.xavier_uniform_(self.fc1_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.fc2_layer.weight, gain=1.414)
        nn.init.xavier_uniform_(self.conv_layer.weight, gain=1.414)

        self.non_linearity = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

        self.to(dev)
        self.dev = dev

    def forward(self, edge_idx, edge_type):
        row, col = edge_idx

        conv_input = torch.stack(
            [self.x[row, :], self.g[edge_type, :], self.g[col, :], self.x_initial[row, :], self.g_initial[edge_type, :],
             self.x_initial[col, :]],
            dim=1).to(self.dev)

        batch_size, length, dim = conv_input.size()
        # assuming inputs are of the form ->
        conv_input = conv_input.transpose(1, 2)
        # batch * length(which is 3 here -> entity,relation,entity) * dim
        # To make tensor of size 4, where second dim is for input channels
        conv_input = conv_input.unsqueeze(1)

        out_conv = self.dropout(
            self.non_linearity(self.conv_layer(conv_input)))

        input_fc = out_conv.squeeze(-1).view(batch_size, -1)
        output = self.fc1_layer(input_fc)
        output = self.fc2_layer(output)
        return output
