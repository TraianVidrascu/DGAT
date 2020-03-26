import torch

from data.dataset import FB15Dataset


class Stats:

    def __init__(self, dataset, dev='cpu'):
        self.dataset = dataset
        self.h, self.g = dataset.load_embedding(dev)

    def get_mean(self, emb):
        return emb.mean()

    def get_variation(self, emb):
        mean = self.get_mean(emb)
        n = emb.shape[0]
        sigma = (emb - mean) ** 2
        sigma = sigma.sum() / n
        return sigma

    def get_std(self, emb):
        sigma = self.get_variation(emb)
        sigma_std = torch.sqrt(sigma)
        return sigma_std

    def get_stats(self):
        mean_h = self.get_mean(self.h)
        sigma_h = self.get_variation(self.h)
        sigma_std_h = self.get_std(self.h)

        mean_g = self.get_mean(self.g)
        sigma_g = self.get_variation(self.g)
        sigma_std_g = self.get_std(self.g)

        return mean_h, sigma_h, sigma_std_h, mean_g, sigma_g, sigma_std_g

    def show_tensor(self, emb):
        print(emb.sum(dim=1))


dataset = FB15Dataset()

sts = Stats(dataset)

results = sts.get_stats()
results = list(map(lambda x: x.item(), results))

result_str = ''
for res in results:
    line = '%.4f' % res + '\n'
    result_str += line

print(result_str)

h, g = dataset.load_embedding()
sts.show_tensor(h)
