import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_mean

from dissimilarities import Dissimilarity, TransE
from relation_embedding_layer import RelationLayer, SimpleRelationLayer


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
        if self.concat:
            h = self._concat(h)
        h = F.normalize(h, dim=-1, p=2)
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
        # h = F.elu(h)
        return h

    def init_params(self):
        nn.init.xavier_normal_(self.weight_inbound.weight, gain=1.414)
        nn.init.xavier_normal_(self.weight_outbound.weight, gain=1.414)
        # nn.init.xavier_normal_(self.lambda_layer.weight, gain=1.414)

        nn.init.zeros_(self.weight_inbound.bias)
        nn.init.zeros_(self.weight_outbound.bias)
        # nn.init.zeros_(self.lambda_layer.bias)


class KB(nn.Module):
    def __init__(self, x, g, backprop_relation, backprop_entity, dissimilarity: Dissimilarity,
                 dev='cpu'):
        super(KB, self).__init__()
        self.x_size = x.shape[1]
        self.g_size = g.shape[1]
        self.n = x.shape[0]
        self.m = g.shape[0]
        self.hidden_size = g.shape[0]

        self.x_initial = nn.Parameter(x, requires_grad=backprop_entity)
        self.g_initial = nn.Parameter(g, requires_grad=backprop_relation)

        self._dissimilarity = dissimilarity
        self.to(dev)

    def loss(self, h_prime, g_prime, pos_edge_idx, pos_edge_type, neg_edge_idx, neg_edge_type):
        # Margin loss
        d_pos = self._dissimilarity(h_prime, g_prime, pos_edge_idx, pos_edge_type)
        d_neg = self._dissimilarity(h_prime, g_prime, neg_edge_idx, neg_edge_type)
        y = torch.ones(d_pos.shape[0]).to(d_pos.device)
        loss = self.loss_fct(d_pos, d_neg, y)
        return loss

    def evaluate(self, h, g, eval_idx, eval_type):
        with torch.no_grad():
            self.eval()
            n = eval_idx.shape[0]
            if n > 15000:
                step = n // 4
                scores = []
                for i in range(0, n, step):
                    batch_idx, batch_type = eval_idx[:, i:i + step], eval_type[i:i + step]

                    preds = torch.detach(self._dissimilarity(h, g, batch_idx, batch_type).cpu())
                    scores.append(preds)
                scores = torch.cat(scores)
            else:
                scores = torch.detach(self._dissimilarity(h, g, eval_idx, eval_type).cpu())
            return scores


class DKBATNet(KB):
    def __init__(self, x, g, output_size, heads, margin=1, dropout=0.3,
                 negative_slope=0.2,
                 use_simple_relation=True,
                 backprop_entity=True,
                 backprop_relation=True,
                 dissimilarity: Dissimilarity = TransE(),
                 device='cpu'):
        super(DKBATNet, self).__init__(x, g,  backprop_relation, backprop_entity, dissimilarity,
                                       device)
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
        if use_simple_relation:
            self.relation_layer = SimpleRelationLayer(self.g_size, output_size * heads, device=device)
        else:
            self.relation_layer = RelationLayer(self.g_size, output_size * heads, device=device)

        self.dropout = nn.Dropout(dropout)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size
        self.use_simple_relation = use_simple_relation

        self.to(device)

    def forward(self, edge_idx, edge_type, path_idx, path_type, use_path):
        torch.cuda.empty_cache()
        self.x_initial.data = F.normalize(
            self.x_initial.data, p=2, dim=1)
        self.g_initial.data = F.normalize(self.g_initial.data, p=2, dim=1)
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
        if self.use_simple_relation:
            g_prime = self.relation_layer(self.g_initial)
        else:
            if use_path:
                g_prime = self.relation_layer(self.g_initial, h_ijk[:edge_type.shape[0], :], edge_type)
            else:
                g_prime = self.relation_layer(self.g_initial, h_ijk, edge_type)

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
        g_prime = F.normalize(g_prime, p=2, dim=1)
        return h_prime, g_prime


class KBNet(KB):
    def __init__(self, x, g, output_size, heads, margin=1, dropout=0.3, negative_slope=0.2,
                 use_simple_relation=True,
                 backprop_entity=True,
                 backprop_relation=True,
                 dissimilarity: Dissimilarity = TransE(),
                 device='cpu'):
        super(KBNet, self).__init__(x, g, backprop_relation, backprop_entity, dissimilarity,
                                    device)
        self.input_attention_layer = RelationalAttentionLayer(self.x_size, self.g_size, output_size, heads,
                                                              negative_slope=negative_slope,
                                                              dropout=dropout,
                                                              device=device)
        self.output_attention_layer = RelationalAttentionLayer(output_size * heads, output_size * heads,
                                                               output_size * heads, 1, dropout=dropout,
                                                               negative_slope=negative_slope,
                                                               device=device, concat=False)

        self.entity_layer = EntityLayer(self.x_size, heads * output_size, device=device)
        if use_simple_relation:
            self.relation_layer = SimpleRelationLayer(self.g_size, output_size * heads, device=device)
        else:
            self.relation_layer = RelationLayer(self.g_size, output_size * heads, device=device)

        self.loss_fct = nn.MarginRankingLoss(margin=margin)

        self.heads = heads
        self.output_size = output_size
        self.use_simple_relation = use_simple_relation

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

        # compute relation embedding
        if self.use_simple_relation:
            g_prime = self.relation_layer(self.g_initial)
        else:
            if use_path:
                g_prime = self.relation_layer(self.g_initial, h_ijk[:edge_type.shape[0], :], edge_type)
            else:
                g_prime = self.relation_layer(self.g_initial, h_ijk, edge_type)

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
