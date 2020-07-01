from data.dataset import WN18RR
from dataloader import DataLoader

dataset = WN18RR()
data_dir = './data/WN18RR/simple_merge'

ldr = DataLoader(dataset)

ldr.load