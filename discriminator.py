import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter_add

from torch.distributions import Normal


class Discriminator(nn.Module):
    def __init__(self, x_size, g_size, embedding_size, hidden_size, dev='cpu'):
        super(Discriminator, self).__init__()
        self.fc1_entity = nn.Linear(embedding_size, hidden_size, bias=True)
        self.fc2_entity = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3_entity = nn.Linear(hidden_size, 1, bias=True)

        self.fc1_rel = nn.Linear(embedding_size, hidden_size, bias=True)
        self.fc2_rel = nn.Linear(hidden_size, hidden_size, bias=True)
        self.fc3_rel = nn.Linear(hidden_size, 1, bias=True)

        self.dev = dev
        self.init_params()

        self.x_size = x_size
        self.g_size = g_size
        self.embedding_size = embedding_size

        self.distribution = Normal(torch.zeros(self.embedding_size).float(), torch.ones(self.embedding_size).float())

        self.loss_fct = nn.BCELoss()
        self.to(dev)

    def init_params(self):
        nn.init.xavier_normal_(self.fc1_entity.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2_entity.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc3_entity.weight, gain=1.414)

        nn.init.xavier_normal_(self.fc1_rel.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc2_rel.weight, gain=1.414)
        nn.init.xavier_normal_(self.fc3_rel.weight, gain=1.414)

        nn.init.zeros_(self.fc1_entity.bias)
        nn.init.zeros_(self.fc2_entity.bias)
        nn.init.zeros_(self.fc3_entity.bias)

        nn.init.zeros_(self.fc1_rel.bias)
        nn.init.zeros_(self.fc2_rel.bias)
        nn.init.zeros_(self.fc3_rel.bias)

    def pred_entity(self, h):
        h = self.fc1_entity(h)
        h = F.elu(h)
        # h = self.fc2_entity(h)
        # h = F.elu(h)
        h = self.fc3_entity(h)
        pred_h = torch.sigmoid(h)
        return pred_h.squeeze()

    def pred_rel(self, g):
        g = self.fc1_entity(g)
        g = F.elu(g)
        # g = self.fc2_entity(g)
        # g = F.elu(g)
        g = self.fc3_entity(g)
        pred_g = torch.sigmoid(g)
        return pred_g.squeeze()

    def forward(self, h, g):
        h = self.fc1_entity(h)
        h = F.elu(h)
        # h = self.fc2_entity(h)
        # h = F.elu(h)
        h = self.fc3_entity(h)
        pred_h = torch.sigmoid(h)

        g = self.fc1_entity(g)
        g = F.elu(g)
        # g = self.fc2_entity(g)
        # g = F.elu(g)
        g = self.fc3_entity(g)
        pred_g = torch.sigmoid(g)

        return pred_h.squeeze(), pred_g.squeeze()

    def loss(self, preds, y):
        loss_value = self.loss_fct(preds.squeeze(), y)
        return loss_value

    def real_loss(self, preds_real):
        y = torch.ones(preds_real.shape[0]).to(self.dev)
        loss = self.loss(preds_real, y)
        return loss

    def loss_entity_embeddings(self, h, edge_idx, edge_type):
        z = self.sample_gaussian(h.shape[0])
        preds_real = self.pred_entity(z)
        preds_fake = self.pred_entity(h.detach())
        loss_real = self.real_loss(preds_real)
        loss_fake = self.fake_loss(preds_fake)
        return (loss_real + loss_fake) / 2

    def loss_relation_embeddings(self, g, edge_idx, edge_type):
        z = self.sample_gaussian(g.shape[0])
        preds_real = self.pred_rel(z)
        preds_fake = self.pred_rel(g.detach())
        loss_real = self.real_loss(preds_real)
        loss_fake = self.fake_loss(preds_fake)
        return (loss_real + loss_fake) / 2

    def fake_loss(self, preds_fake):
        y = torch.zeros(preds_fake.shape[0]).to(self.dev)
        loss = self.loss(preds_fake, y)
        return loss

    def discriminator_loss(self, h, g, edge_idx, edge_type):
        loss_g = self.loss_relation_embeddings(g, edge_idx, edge_type)
        loss_h = self.loss_entity_embeddings(h, edge_idx, edge_type)
        return (loss_g + loss_h) / 2

    def sample_gaussian(self,n):
        z = self.distribution.sample((n,)).to(self.dev)
        return z.detach()

    def adversarial_loss(self, preds_fake):
        y = torch.ones(preds_fake.shape[0]).to(self.dev)
        loss = self.loss(preds_fake, y)
        return loss

    def adversarial_loss_entities(self, h):
        preds_fake = self.pred_entity(h)
        loss = self.adversarial_loss(preds_fake)
        return loss

    def adversarial_loss_relations(self, g):
        preds_fake = self.pred_rel(g)
        loss = self.adversarial_loss(preds_fake)
        return loss
