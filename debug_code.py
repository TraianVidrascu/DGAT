# import time
# import torch

# from torch_scatter import scatter_add
# import torch_sparse
#
# from data.dataset import FB15Dataset
# from dataloader import DataLoader
#
# t = time.time()
# dataset_a = FB15Dataset()
# s = time.time()
# print('Initialize Dataset:' + str(s - t))
# #
# t = time.time()
# #dataset_a.pre_process()
# s = time.time()
# print('Process Dataset:' + str(s - t))
#
# t = time.time()
# ldr = DataLoader(dataset_a)
# s = time.time()
# print('Load Dataset:' + str(s - t))
# #
# t = time.time()
# x_a, g_a, graph_a = ldr.load_train()
# s = time.time()
# print('Load Train Fold:' + str(s - t))
#
# t = time.time()
# graph_reduced_a, edge_idx_unseen_a, edge_lbl_unseen_a = ldr.split_unseen(graph_a)
# s = time.time()
# print('Split Unseen Edges:' + str(s - t))
#
# t = time.time()
# e_idx, e_lbl = ldr.graph2idx(graph_a)
# s = time.time()
# print('Convert to idx:' + str(s - t))
#
# t = time.time()
# n = len(graph_a)
# ldr.negative_samples(n, e_idx, e_lbl)
# s = time.time()
# print('Get negative Samples:' + str(s - t))

# n = 27000
# m = 250000
# edge_idx = torch.randint(n, size=(2, m)).long()
# t = time.time()
# matrix = torch.sparse_coo_tensor(edge_idx, torch.ones(m), size=(n, n))
# s = time.time()
# print('Sparse matrix creation time:' + str(s - t))
#
# r, c = edge_idx
# matrix[r,c] = 0
# m = torch.rand(size=(m))
# t = time.time()
# r, c = edge_idx
# matrix[r,c] = m
# s = time.time()

# k, n, m, d, h, heads = 237, 27000, 280000, 300, 200, 2
#
# x = torch.rand(size=(n, d))
# g = torch.rand(size=(k, d))
#
# edge_idx = torch.randint(high=n, size=(2, m))
# edge_type = torch.randint(high=k, size=[m])
#
# fc1 = torch.nn.Linear(3 * d, heads * h)
# att_weights = torch.rand((1, heads, h))
# att_actv = torch.nn.LeakyReLU()
#
#
# # print('Softmax computation time:' + str(e - s))
#
# def _sofmax(indexes, edge_values):
#     """
#
#     :param indexes: nodes of each edge
#     :param egde_values: values of each edge
#     :return: normalized values of edges considering nodes
#     """
#     edge_values = torch.exp(edge_values).t()
#
#     # check if correct?
#     row_sum = scatter_add(edge_values, indexes)
#     if len(row_sum.shape) > 1:
#         edge_values /= row_sum[:, indexes]
#     else:
#         edge_values /= row_sum[indexes]
#
#     return edge_values
#
#
# def _compute_edges(edge_idx, edge_type, x, g):
#     row, col = edge_idx
#
#     # extract embeddings for entities and relations
#     h_i = x[row]
#     h_j = x[col]
#     g_k = g[edge_type]
#
#     # concatenate the 3 representations
#     h_ijk = torch.cat([h_i, h_j, g_k], dim=1)
#     # obtain edge representation
#     c_ijk = fc1(h_ijk).view(-1, heads, h)
#
#     return c_ijk
#
#
# def _attention(c_ijk, edge_idx):
#     a_ijk = torch.sum(att_weights * c_ijk, dim=2)
#     b_ijk = att_actv(a_ijk).squeeze()
#
#     rows = edge_idx[0, :]
#     alpha = _sofmax(rows, b_ijk)
#
#     return alpha
#
#
# def _aggregate(edge_idx, alpha, c_ijk):
#     rows = edge_idx[0, :]
#     a = alpha[None, :, :] * c_ijk.transpose(0, 2)
#     h = scatter_add(a, rows).transpose(0, 2)
#     return h
#
#
# s = time.time()
# c_ijk = _compute_edges(edge_idx, edge_type, x, g)
# t = time.time()
# print(c_ijk.shape)
# print('Edge embedding computation time:' + str(t - s))
#
# s = time.time()
# alpha = _attention(c_ijk, edge_idx)
# t = time.time()
# print(alpha.shape)
# print('Attention computation time:' + str(t - s))
#
# s = time.time()
# h = _aggregate(edge_idx, alpha, c_ijk)
# t = time.time()
# print(h.shape)
# print('Aggregation computation time:' + str(t - s))
#
# s = time.time()
# print(h[0,:,0])
# h /= heads
# h = torch.sum(h, dim=1).squeeze()
# t = time.time()
# print(h[0,0])
#
# print(h.shape)
# print('Concatenation computation time:' + str(t - s))
# epochs = 20
# print("Epoch %4d" % (epochs + 1))

# n = 5
# edge_idx = torch.randint(size=(2, n+1), high=n + 1)
#
# a_in = torch.sparse_coo_tensor(edge_idx, torch.ones(n+1), size=(n + 1, n + 1))
# print('Input adj matrix')
# print(a_in.to_dense())
# print('Transpose adj matrix')
# print(a_in.t())
# print('Output adj')
# edge_idx_out = torch.stack([edge_idx[1,:],edge_idx[0,:]])
# a_out = torch.sparse_coo_tensor(edge_idx_out, torch.ones(n+1), size=(n + 1, n + 1))
# print(a_out.to_dense())
# print('Is Equal')
# print(a_out.to_dense() == a_in.to_dense().t())
#
#
# print(a_in.t())
# print(a_out)
# print(a_in)

# n = 10
# channels = 3
# input_size = 4
#
# emb = torch.tensor([i for i in range(1, input_size * 3 + 1)]).view(1, 3, input_size).float()
# print(emb)
# print(emb.shape)
# conv = torch.nn.Conv1d(3, channels, 1)
# torch.nn.Parameter(torch.Tensor(3, channels))
# res = conv(emb)
# print(res)
# print(res.shape)

# file_h = './data/FB15k-237/processed/h.pt'
# file_g = './data/FB15k-237/processed/g.pt'
# h = torch.load(file_h)
# g = torch.load(file_g)
#
# a = torch.randperm(100)
# batch_size = 30
# for i in range(0,100,batch_size):
#     print(i)
#     j = i+batch_size
#     print(j)
#     print(a[i:j])

# preds = torch.tensor([0, 1, 1e-20, 1e-20, 1e-20, 1e+1, 1e+1, 1e+1])
# target = torch.tensor([-1 for i in range(preds.shape[0])]).float()
#
# loss = torch.nn.SoftMarginLoss(reduction='none')
# print(preds)
# print(target)
# res = loss(preds, target)
# print(res)

# power = torch.tensor([-i for i in range(1000)]).float()
# print(power)
# print(torch.exp(power))
# res =torch.log(1+ torch.exp(power))
# print(res)
import time

import torch

from data.dataset import FB15Dataset
from dataloader import DataLoader

# dataset = FB15Dataset()
# ldr = DataLoader(dataset)
# n, k = ldr.get_properties()
# triplet = torch.tensor([[1], [2]]).long()
# print(triplet)
# print(triplet.shape)
# triplets, position = ldr.corrupt_triplet(n, triplet)
#
# print(triplets)
# print(triplets.shape)
# print(triplets[:, position])
#
# triplets, position = ldr.corrupt_triplet(n, triplet,head=False)
#
# print(triplets)
# print(triplets.shape)
# print(triplets[:, position])

row = torch.tensor([1,2,3,4])
print(row)
m = row.shape[0]
print(row.expand(4,2))