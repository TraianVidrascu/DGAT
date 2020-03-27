import argparse

import torch
import numpy as np
import wandb

from data.dataset import FB15Dataset
from dataloader import DataLoader
from metrics import get_metrics, rank_triplet
from model import KBNet
from utilis import save_best, load_model, set_random_seed

ENCODER_FILE = 'encoder.pt'
ENCODER_CHECKPOINT = 'encoder_check.pt'


def get_encoder(args, x_size, g_size):
    # model parameters
    h_size = args.hidden_encoder
    o_size = args.output_encoder
    heads = args.heads
    margin = args.margin

    dev = args.device

    model = KBNet(x_size, g_size, h_size, o_size, heads, margin, device=dev)

    return model


def train_encoder(args, model, data_loader):
    # system parameters
    dev = args.device
    eval = args.eval
    # training parameters
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    dataset_name = data_loader.get_name()

    # load data
    x, g, graph = data_loader.load_train(dev)

    wandb.watch(model, log="all")

    first = 0
    # if args.checkpoint:
    #     model, first = load_model(model, ENCODER_CHECKPOINT)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)

    pos_edge_idx, edge_type = data_loader.graph2idx(graph, dev)
    n = x.shape[0]

    for epoch in range(first, epochs):
        model.train()

        neg_edge_idx = data_loader.negative_samples(n, pos_edge_idx, dev)

        h_prime, g_prime = model(x, g, pos_edge_idx, edge_type)

        loss = model.loss(h_prime, g_prime, pos_edge_idx, neg_edge_idx, edge_type)

        # optimization
        optim.zero_grad()
        loss.backward()
        optim.step()

        # save_model(model, loss.item(), epoch + 1, ENCODER_CHECKPOINT)
        save_best(model, loss.item(), epoch + 1, ENCODER_FILE, asc=False)

        if (epoch + 1) % eval == 0:
            metrics = get_encoder_metrics(h_prime, g_prime, data_loader, 'valid', model, dev=args.device)
            metrics['train_' + dataset_name + '_Loss_encoder'] = loss
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_encoder': loss})

        del neg_edge_idx, h_prime, g_prime, loss
        torch.cuda.empty_cache()

    del pos_edge_idx, edge_type, x, g, graph, model
    torch.cuda.empty_cache()


def evaluate_encoder(h, g, dataloader, fold, score_fct, dev='cpu'):
    with torch.no_grad():
        n, k = dataloader.get_properties()
        _, _, graph = dataloader.load(fold, dev)
        edge_idx, edge_type = dataloader.graph2idx(graph, dev)
        m = edge_idx.shape[1]

        ranks_head = []

        for i in range(m):
            triplet_idx = edge_idx[:, i]
            triplet_type = edge_type[i]

            triplets_head, position_head = dataloader.corrupt_triplet(n, triplet_idx)

            triplets_head_type = torch.zeros(triplets_head.shape[1]).long()
            triplets_head_type[:] = triplet_type
            scores_head = score_fct(h, g, triplets_head, triplets_head_type).cpu().numpy()

            rank_head = rank_triplet(scores_head, position_head)
            ranks_head.append(rank_head)
        ranks_head = np.array(ranks_head)

        return ranks_head


def get_encoder_metrics(h, g, data_loader, fold, encoder, dev='cpu'):
    ranks = evaluate_encoder(h, g, data_loader, fold, encoder._dissimilarity, dev=dev)
    mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranks)

    dataset = data_loader.get_name()

    metrics = {fold + '_' + dataset + '_MR_encoder': mr,
               fold + '_' + dataset + '_MRR_encoder': mrr,
               fold + '_' + dataset + '_Hits@1_encoder': hits_1,
               fold + '_' + dataset + '_Hits@3_encoder': hits_3,
               fold + '_' + dataset + '_Hits@10_encoder': hits_10}
    return metrics


def embed_nodes(args, encoder, data):
    dev = args.device

    data_loader = DataLoader(data)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, dev)

    encoder.eval()
    with torch.no_grad():
        h, g = encoder(x, g, edge_idx, edge_type)
    return h, g


def load_encoder(args, g_size, x_size):
    encoder = get_encoder(args, x_size, g_size)
    encoder, _ = load_model(encoder, ENCODER_FILE)
    return encoder


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=100, help="After how many epochs to evaluate.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs for encoder.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay encoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training.")
    parser.add_argument("--dataset", type=str, default='FB15k-237', help="Dataset used for training.")

    # objective function parameters
    parser.add_argument("--margin", type=int, default=1, help="Margin for loss function.")

    # encoder parameters
    parser.add_argument("--negative_slope", type=float, default=0.2, help="Negative slope for Leaky Relu")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads per layer")
    parser.add_argument("--hidden_encoder", type=int, default=200, help="Number of neurons per hidden layer")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    wandb.init(project="KBAT_encoder", config=args)

    dataset = FB15Dataset()
    data_loader = DataLoader(dataset)

    # load model architecture
    x_size = dataset.size_x
    g_size = dataset.size_g
    model = get_encoder(args, x_size, g_size)

    # initial evaluation for test and valid folds
    x, g, _ = data_loader.load('valid')  # fold does not matter as we are only interested in initial embedding
    metrics = get_encoder_metrics(x, g, data_loader, 'test', model, dev=args.device)
    wandb.log(metrics)
    metrics = get_encoder_metrics(x, g, data_loader, 'valid', model, dev=args.device)
    wandb.log(metrics)

    # train model and save embeddings
    train_encoder(args, model, data_loader)
    h, g = embed_nodes(args, model, dataset)
    dataset.save_embedding(h, g)

    # evaluate test and valid fold after training
    metrics = get_encoder_metrics(h, g, data_loader, 'valid', model, dev=args.device)
    wandb.log(metrics)
    metrics = get_encoder_metrics(h, g, data_loader, 'test', model, dev=args.device)
    wandb.log(metrics)


if __name__ == "__main__":
    main()
