import torch
import torch.nn as nn
import torch.nn.functional as F


class WrapperConvKB(nn.Module):
    def __init__(self, h, g, embedding_size, out_channels, drop_prob=0.0, dev='cpu'):
        super(WrapperConvKB, self).__init__()
        self.conv = ConvE(embedding_size, out_channels)

        self.node_embeddings = nn.Parameter(h, requires_grad=True)
        self.rel_embeddings = nn.Parameter(g, requires_grad=True)

        self.dev = dev
        self.out_channels = out_channels
        self.to(dev)

    def forward(self, edge_idx, edge_type):
        preds = self.conv(self.node_embeddings, self.rel_embeddings, edge_idx, edge_type)
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


class ConvE(nn.Module):
    def __init__(self, embedding_size, out_channels, input_drop=0.2, hidden_drop=0.2, feat_drop=0.3):
        super(ConvE, self).__init__()
        self.hidden_size = embedding_size
        self.out_channels = out_channels

        self.conv = nn.Conv2d(1, out_channels, kernel_size=(2, 1),
                              bias=True)
        self.fc1 = nn.Linear(out_channels * embedding_size, embedding_size)

        self.bn1 = nn.BatchNorm2d(1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm1d(embedding_size)

        self.input_drop = torch.nn.Dropout(input_drop)
        self.hidden_drop = torch.nn.Dropout(hidden_drop)
        self.feature_map_drop = torch.nn.Dropout2d(feat_drop)

    def forward(self, h, g, edge_idx, edge_type):
        row, col = edge_idx
        rel = edge_type

        c_ik = torch.stack([h[row, :], g[rel, :]], dim=1).unsqueeze(1)
        self.input_drop(c_ik)
        c_ik = self.bn1(c_ik)
        c_j = h[col, :]
        h = self.conv(c_ik)
        h = self.feature_map_drop(h)
        self.bn2(h)
        h = F.relu(h)
        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = self.hidden_drop(h)
        h = self.bn3(h)
        h = F.relu(h)
        h = torch.sum(h * c_j, dim=1).unsqueeze(-1)
        return h
