import numpy as np
import torch

from dataloader import DataLoader


def rank_triplet(scores, position):
    sorted_scores, sorted_indices = torch.sort(
        scores.view(-1), dim=-1, descending=True)
    rank = (sorted_indices == position).nonzero() + 1
    return rank


def mean_reciprocal_rank(ranks):
    tensor = torch.FloatTensor(ranks)
    tensor = 1 / tensor
    mrr = torch.sum(tensor) / len(ranks)
    return mrr.item()


def mean_rank(ranks):
    return sum(ranks).item() / len(ranks)


def hits_at(ranks):
    hits_1 = 0
    hits_3 = 0
    hits_10 = 0
    for i in range(len(ranks)):
        hits_1 += 1 if ranks[i] == 1 else 0
        hits_3 += 1 if ranks[i] <= 3 else 0
        hits_10 += 1 if ranks[i] <= 10 else 0
    hits_1 /= len(ranks)
    hits_3 /= len(ranks)
    hits_10 /= len(ranks)
    return hits_1, hits_3, hits_10


def get_metrics(ranks):
    mr = mean_rank(ranks)
    mrr = mean_reciprocal_rank(ranks)
    hits_1, hits_3, hits_10 = hits_at(ranks)

    return mr, mrr, hits_1, hits_3, hits_10


def get_ranking_metric(ranking_name, ranking, model_name, dataset_name, fold):
    mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranking)

    metrics = {fold + '_' + dataset_name + '_' + ranking_name + '_MR_' + model_name: mr,
               fold + '_' + dataset_name + '_' + ranking_name + '_MRR_' + model_name: mrr,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@1_' + model_name: hits_1,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@3_' + model_name: hits_3,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@10_' + model_name: hits_10}
    return metrics


def get_model_metrics(data_loader, h, g, fold, model, model_name, dev='cpu'):
    ranks_head = evaluate_filtered(model, h, g, data_loader, fold, True, dev)
    ranks_tail = evaluate_filtered(model, h, g, data_loader, fold, False, dev)
    ranks = np.concatenate((ranks_head, ranks_tail))

    dataset_name = data_loader.get_name()

    metrics_head = get_ranking_metric('head', ranks_head, dataset_name, model_name, fold)
    metrics_tail = get_ranking_metric('tail', ranks_tail, dataset_name, model_name, fold)
    metrics_all = get_ranking_metric('both', ranks, dataset_name, model_name, fold)

    metrics = {**metrics_head, **metrics_tail, **metrics_all}
    return metrics


def get_model_metrics_head_or_tail(data_loader, h, g, fold, model, model_name, head, dev='cpu'):
    ranks = evaluate_filtered(model, h, g, data_loader, fold, head, dev)

    dataset_name = data_loader.get_name()
    part = 'head' if head else 'tail'
    metrics = get_ranking_metric(part, ranks, dataset_name, model_name, fold)

    return metrics


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


def evaluate_filtered(model, h, g, data_loader, fold, head, dev='cpu'):
    with torch.no_grad():
        # load structural information

        # load corrupted head triplets

        triplets_file = data_loader.get_filtered_eval_file(fold, head)
        ranks = []
        counter = 0
        while True:
            eval_idx, eval_type, position = DataLoader.load_list(triplets_file, head, dev)

            # if no more lists break
            if eval_idx is None:
                break
            scores = model.evaluate(h, g, eval_idx.to(dev), eval_type.to(dev))
            rank = rank_triplet(scores, position)
            ranks.append(rank)
            torch.cuda.empty_cache()

            # remove it when runnig experiments, only for debug
            counter += 1
            print("List %.d" % counter + ' rank: %.d' % rank.item())
        triplets_file.close()
        torch.cuda.empty_cache()
        return ranks
