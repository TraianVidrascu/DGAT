import numpy as np
import pandas as pd


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
    hits = hits_at(ranks, 10)
    return mr, mrr, hits
