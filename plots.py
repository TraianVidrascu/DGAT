from matplotlib import pyplot as plt
import os.path as osp

data_dir = './data/WN18RR/analysis/'

entity_initial_path = osp.join(data_dir, 'initial/2DPCA_initial_entity.svg')

entity_initial = plt.imread(entity_initial_path)

fig, ax = plt.subplots()
im = ax.imshow(entity_initial)
plt.show()
