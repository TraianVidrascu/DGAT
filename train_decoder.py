import argparse
import concurrent.futures

import torch
import numpy as np
import wandb

from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import rank_triplet, get_metrics, evaluate
from model import ConvKB
from utilis import load_model, save_model, save_best, set_random_seed

DECODER_FILE = 'decoder.pt'
DECODER_CHECKPOINT = 'decoder_checkpoint.pt'


def get_decoder(args):
    channels = args.channels
    dropout = args.dropout
    input_size = args.output_encoder
    dev = args.device

    model = ConvKB(input_size, channels, dropout=dropout, dev=dev)
    return model


def train_decoder(args, decoder, data_loader):
    wandb.watch(decoder, log="all")

    dataset_name = data_loader.get_name()

    dev = args.device
    negative_ratio = args.negative_ratio
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    eval = args.eval
    batch_size = args.batch_size

    _, _, graph = data_loader.load_train(dev)
    h, g = data_loader.load_embedding(dev)

    first = 0
    # if args.checkpoint:
    #     decoder, first = load_model(decoder, DECODER_CHECKPOINT)

    optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=25, gamma=0.5, last_epoch=-1)

    pos_edge_idx, pos_edge_type = DataLoader.graph2idx(graph, dev)
    m = pos_edge_idx.shape[1]
    n = h.shape[0]

    for epoch in range(first, epochs):
        decoder.train()
        neg_edge_idx, neg_edge_type = data_loader.negative_samples(n, pos_edge_idx, pos_edge_type, negative_ratio, dev)

        m_neg = neg_edge_idx.shape[1]

        # input and target ( positive and negative samples
        target = torch.cat([torch.ones(m), -torch.ones(m_neg)]).to(dev)
        edge_idx = torch.cat([pos_edge_idx, neg_edge_idx], dim=1)
        edge_type = torch.cat([pos_edge_type, neg_edge_type])

        # shuffle pos and negative samples
        edge_idx, edge_type = data_loader.shuffle_samples(edge_idx, edge_type)
        total_size = edge_idx.shape[1]

        iteration = torch.randperm(total_size).to(dev)
        loss_epoch = 0
        no_batch = total_size / batch_size
        for itt in range(0, total_size, batch_size):
            batch = iteration[itt:itt + batch_size]

            batch_idx = edge_idx[:, batch]
            batch_type = edge_type[batch]
            batch_target = target[batch]

            prediction = decoder(h, g, batch_idx, batch_type)

            loss = decoder.loss(prediction, batch_target)

            # optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += loss.item() / no_batch

        scheduler.step()

        # save_model(decoder, loss_epoch, epoch + 1, DECODER_CHECKPOINT)
        save_best(decoder, loss_epoch, epoch + 1, DECODER_FILE, asc=False)

        if (epoch + 1) % eval == 0:
            metrics = get_decoder_metrics(decoder, data_loader, 'valid', dev)

            metrics['train_' + dataset_name + '_Loss_decoder'] = loss_epoch
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_decoder': loss_epoch})


def load_decoder(args, dev='cuda'):
    decoder = get_decoder(args, dev)
    model, _ = load_model(decoder, DECODER_FILE)
    return model





def get_ranking_metric(ranking_name, ranking, dataset_name, fold):
    mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranking)

    metrics = {fold + '_' + dataset_name + '_' + ranking_name + '_MR_decoder': mr,
               fold + '_' + dataset_name + '_' + ranking_name + '_MRR_decoder': mrr,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@1_decoder': hits_1,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@3_decoder': hits_3,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@10_decoder': hits_10}
    return metrics


def get_decoder_metrics(model, data_loader, fold, dev='cpu'):
    ranks_head, ranks_tail, ranks = evaluate(model, data_loader, fold, dev=dev)

    dataset_name = data_loader.get_name()

    metrics_head = get_ranking_metric('head', ranks_head, dataset_name, fold)
    metrics_tail = get_ranking_metric('tail', ranks_tail, dataset_name, fold)
    metrics_all = get_ranking_metric('both', ranks, dataset_name, fold)

    metrics = {**metrics_head, **metrics_tail, **metrics_all}
    return metrics


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=1, help="After how many epochs to evaluate.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for decoder.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay decoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training")
    parser.add_argument("--batch_size", type=int, default=8000, help="Batch size for decoder.")
    parser.add_argument("--negative-ratio", type=int, default=2, help="Number of negative samples.")
    parser.add_argument("--dataset", type=str, default='FB15k-237', help="Dataset used for training.")

    # objective function parameters
    parser.add_argument("--margin", type=int, default=1, help="Margin for loss function.")

    # decoder parameters
    parser.add_argument("--channels", type=int, default=50, help="Number of channels for decoder.")
    parser.add_argument("--output_encoder", type=int, default=400, help="Number of neurons per output layer")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    wandb.init(project="KBAT_decoder", config=args)

    # load dataset
    if args.dataset == 'FB15k-237':
        dataset = FB15Dataset()
    else:
        dataset = WN18RR()
    data_loader = DataLoader(dataset)

    # load model architecture
    decoder = get_decoder(args)

    # train decoder model
    train_decoder(args, decoder, data_loader)

    # Evaluate test and valid fold after training is done
    metrics = get_decoder_metrics(decoder, data_loader, 'test', args.device)
    wandb.log(metrics)
    metrics = get_decoder_metrics(decoder, data_loader, 'valid', args.device)
    wandb.log(metrics)


if __name__ == "__main__":
    main()
