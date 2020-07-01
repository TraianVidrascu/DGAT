import string

import torch
from adjustText import adjust_text
from torch_scatter import scatter_add
from mpl_toolkits import mplot3d

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from data.dataset import WN18RR, FB15Dataset, Ivy
from dataloader import DataLoader


def compute_pca(embeddings):
    pca_2 = PCA(n_components=2)
    embeddings_results = pca_2.fit_transform(embeddings.cpu().numpy())
    return embeddings_results


def compute_model_analyses(enitiy, relation, edges):
    pca_entity = compute_pca(enitiy)
    pca_relation = compute_pca(relation)
    pca_edges = compute_pca(edges)
    return pca_entity, pca_relation, pca_edges


def pca2plots(pca_initial, pca_encoder, pca_decoder, space, colors=None, description=None,
              cmap='gist_rainbow', over_plot=False, labels=None):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [6, 6, 7]}, figsize=(30, 10))
    plt.figure()

    if not over_plot:
        ax1.scatter(pca_initial[:, 0], pca_initial[:, 1], c=colors, cmap=cmap, )
        ax2.scatter(pca_encoder[:, 0], pca_encoder[:, 1], c=colors, cmap=cmap)
        sc3 = ax3.scatter(pca_decoder[:, 0], pca_decoder[:, 1], c=colors, cmap=cmap)
    else:
        ax1.scatter(pca_initial[:, 0], pca_initial[:, 1], c=colors, cmap=cmap, marker='.', lw=0, alpha=0.3)
        ax2.scatter(pca_encoder[:, 0], pca_encoder[:, 1], c=colors, cmap=cmap, marker='.', lw=0, alpha=0.3)
        sc3 = ax3.scatter(pca_decoder[:, 0], pca_decoder[:, 1], c=colors, cmap=cmap, marker='.', lw=0, alpha=0.3)

    # set subtitles
    ax1.set_title('Initial', fontsize=24)
    ax2.set_title('Encoder', fontsize=24)
    ax3.set_title('Decoder', fontsize=24)

    cbar = fig.colorbar(sc3)
    cbar.set_label(description, fontsize=24)

    fig.suptitle('2D PCA over ' + space + ' embeddings', fontsize=28)

    if labels is not None:
        texts1 = []
        texts2 = []
        texts3 = []
        for i, label in enumerate(labels):
            txt = ax1.text(pca_initial[i, 0], pca_initial[i, 1], label)
            texts1.append(txt)
            txt = ax2.text(pca_encoder[i, 0], pca_encoder[i, 1], label)
            texts2.append(txt)
            txt = ax3.text(pca_decoder[i, 0], pca_decoder[i, 1], label)
            texts3.append(txt)

        adjust_text(texts1, ax=ax1)
        adjust_text(texts2, ax=ax2)
        adjust_text(texts3, ax=ax3)

    fig.savefig('2DPCA_' + space + '.pdf', format='pdf')
    print('done')


def get_edge_embeddings(h, g, row, rel_type, col):
    h_ijk = torch.cat([h[row, :], h[col, :], g[rel_type, :]], dim=1)
    return h_ijk


def get_degree(n, row, col):
    edges = torch.cat([torch.ones(row.shape[0]).long(), torch.zeros(n).long()], dim=-1)
    col = torch.cat([col, torch.tensor([i for i in range(n)])], dim=-1)
    row = torch.cat([row, torch.tensor([i for i in range(n)])], dim=-1)

    in_deg = scatter_add(edges, col)
    out_deg = scatter_add(edges, row)
    deg = in_deg + out_deg
    return deg.numpy()


def get_rel_frequency(rel):
    edges = torch.ones(rel.shape[0]).long()
    freq = scatter_add(edges, rel)
    return freq


def analyse(pca_initial, pca_encoder, pca_decoder, space, colors, desc, over_plot=False, cmap='gist_rainbow',
            labels=None):
    pca2plots(pca_initial, pca_encoder, pca_decoder, space, colors, description=desc, cmap=cmap, over_plot=over_plot,
              labels=labels)


if __name__ == '__main__':
    dataset = WN18RR()
    data_dir = './data/WN18RR/simple_merge'

    ldr = DataLoader(dataset)
    row, rel_type, col = dataset.get_valid_triplets()
    deg = get_degree(dataset.n, row, col)
    freq = get_rel_frequency(rel_type)

    deg = np.log(deg)
    freq = np.log(freq)

    file_encoder = data_dir + '/encoder_final.pth'
    file_decoder = data_dir + '/decoder_final.pth'

    encoder = torch.load(file_encoder)
    decoder = torch.load(file_decoder)

    # Analyse initial embeddings
    h_initial, g_initial, _ = ldr.load_train()
    edges_initial = get_edge_embeddings(h_initial, h_initial, row, rel_type, col)
    pca_entity_initial, pca_relation_initial, pca_edges_initial = compute_model_analyses(h_initial, g_initial,
                                                                                         edges_initial)

    # Analyse encoder embeddings
    g_encoder = encoder['final_relation_embeddings']
    h_encoder = encoder['final_entity_embeddings']
    edges_encoder = get_edge_embeddings(h_encoder, g_encoder, row, rel_type, col)
    pca_entity_encoder, pca_relation_encoder, pca_edges_encoder = compute_model_analyses(h_encoder, g_encoder,
                                                                                         edges_encoder)

    # Analyse decoder embeddings
    g_decoder = decoder['final_relation_embeddings']
    h_decoder = decoder['final_entity_embeddings']
    edges_decoder = get_edge_embeddings(h_decoder, g_decoder, row, rel_type, col)
    pca_entity_decoder, pca_relation_decoder, pca_edges_decoder = compute_model_analyses(h_decoder, g_decoder,
                                                                                         edges_decoder)

    # get labels
    rel_mapper = dataset.load_mapper()

    analyse(pca_entity_initial, pca_entity_encoder, pca_entity_decoder, 'entity', deg, 'Node degree in log space',
            cmap='RdYlGn',
            over_plot=True)

    labels = []
    for key in rel_mapper.keys():
        label = rel_mapper[key]
        # label = label.replace('java.', '')
        # label = label.replace('definition', 'def')
        labels.append(label)
        print(rel_mapper[key] + ' -> ' + label)

    # analyse(pca_relation_initial, pca_relation_encoder, pca_relation_decoder, 'relation', freq,
    #         'Relation frequency in log space',
    #         cmap='jet',
    #         over_plot=False,
    #         labels=labels
    #         )

    analyse(pca_edges_initial, pca_edges_encoder, pca_edges_decoder, 'edges', rel_type.cpu().numpy(),
            'Discrete relation color mapping',
            over_plot=True)
