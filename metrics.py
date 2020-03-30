import numpy as np
import pandas as pd
import torch
import concurrent.futures


def rank_triplet(scores, position):
    labels = np.zeros_like(scores)
    labels[position] = 1
    entries = np.stack([scores, labels]).transpose()

    df = pd.DataFrame(columns=['score', 'label'], data=entries)
    df = df.sort_values(by=['score'], ascending=False)
    df.index = [i for i in range(1, len(scores) + 1)]
    rank = df[df['label'] == 1].index
    return rank


def mean_reciprocal_rank(ranks):
    reciprocal_ranks = 1 / ranks
    mrr = reciprocal_ranks.mean()
    return mrr


def mean_rank(ranks):
    # ToDo: check if I should give a real rank
    return np.floor(ranks.mean())


def hits_at(ranks, n=1):
    no_ranks = ranks.shape[0]
    hits = np.sum(ranks <= n)
    hits_mean = hits / no_ranks
    return hits_mean


def get_metrics(ranks):
    mr = mean_rank(ranks)
    mrr = mean_reciprocal_rank(ranks)
    hits_1 = hits_at(ranks, 1)
    hits_3 = hits_at(ranks, 3)
    hits_10 = hits_at(ranks, 10)

    return mr, mrr, hits_1, hits_3, hits_10


def evaluate_list(model, h, g, corrupted, list_info, head):
    # evaluate for corrupted list
    n = corrupted.shape[0]
    original = list_info[0].expand(n)
    position = list_info[1].item()
    edge_types = list_info[2].expand(n)

    edge_idx = torch.stack([corrupted, original]) if head else torch.stack([original, corrupted])
    scores = model.evaluate(h, g, edge_idx, edge_types)

    rank = rank_triplet(scores, position)
    del n, original, position, edge_types, edge_idx, scores
    torch.cuda.empty_cache()
    return rank


def evaluate(model, dataloader, fold, dev='cpu'):
    with torch.no_grad():
        # load corrupted head triplets
        triplets_head, lists_head = dataloader.load_evaluation_triplets_raw(fold=fold, head=True, dev='cpu')
        # load corrupted tail triplets
        triplets_tail, lists_tail = dataloader.load_evaluation_triplets_raw(fold=fold, head=False, dev='cpu')
        # load relation and node embeddings
        h, g = dataloader.load_embedding('cpu')

        no_lists = triplets_head.shape[0]
        ranks_head = []
        ranks_tail = []

        for list_idx in range(no_lists):
            # evaluate for corrupted head triplets

            rank_head = evaluate_list(model, h.to(dev), g.to(dev), triplets_head[list_idx, :].to(dev),
                                      lists_head[:, list_idx].to(dev), True)
            # evaluate for corrupted tail triplets
            rank_tail = evaluate_list(model, h.to(dev), g.to(dev), triplets_tail[list_idx, :].to(dev),
                                      lists_tail[:, list_idx].to(dev), False)

            ranks_head.append(rank_head)
            ranks_tail.append(rank_tail)

        ranks = ranks_head + ranks_tail

        # convert to numpy arrays
        ranks_head = np.array(ranks_head)
        ranks_tail = np.array(ranks_tail)
        ranks = np.array(ranks)

        return ranks_head, ranks_tail, ranks
