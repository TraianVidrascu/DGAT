import argparse

import torch
import numpy as np
import wandb

from data.dataset import FB15Dataset
from dataloader import DataLoader
from metrics import rank_triplet, get_metrics
from model import ConvKB
from utilis import load_model, save_model, save_best, set_random_seed

DECODER_FILE = 'decoder.pt'
DECODER_CHECKPOINT = 'decoder_checkpoint.pt'


def get_decoder(args):
    channels = args.channels
    dropout = args.dropout
    dev = args.device
    input_size = args.output_encoder

    model = ConvKB(input_size, channels, dropout=dropout, dev=dev)
    return model


def train_decoder(args, data_loader):
    dev = args.device
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    eval = args.eval
    batch_size = args.batch_size

    _, _, graph = data_loader.load_train(dev)
    h, g = data_loader.load_embedding(dev)

    decoder = get_decoder(args)
    wandb.watch(decoder, log="all")

    first = 0
    # if args.checkpoint:
    #     decoder, first = load_model(decoder, DECODER_CHECKPOINT)

    optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=decay)

    pos_edge_idx, edge_type = data_loader.graph2idx(graph, dev)
    m = pos_edge_idx.shape[1]
    n = h.shape[0]
    total_size = m * 2
    edge_type_all = torch.cat([edge_type, edge_type])

    for epoch in range(first, epochs):
        decoder.train()
        neg_edge_idx = data_loader.negative_samples(n, pos_edge_idx, dev)

        target = torch.cat([torch.ones(m), -torch.ones(m)]).to(dev)
        edge_idx = torch.cat([pos_edge_idx, neg_edge_idx], dim=1)

        iteration = torch.randperm(total_size).to(dev)
        loss_epoch = 0
        for itt in range(0, total_size, batch_size):
            batch = iteration[itt:itt + batch_size]

            batch_idx = edge_idx[:, batch]
            batch_type = edge_type_all[batch]
            batch_target = target[batch]

            prediction = decoder(h, g, batch_idx, batch_type)

            loss = decoder.loss(prediction, batch_target)

            # optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += loss.item()

        loss_epoch /= (total_size / batch_size)
        # save_model(decoder, loss_epoch, epoch + 1, DECODER_CHECKPOINT)
        save_best(decoder, loss_epoch, epoch + 1, DECODER_FILE, asc=False)

        if (epoch + 1) % eval == 0:
            ranks = evaluate_decoder(decoder, data_loader, 'valid', dev=args.device)

            mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranks)
            wandb.log({'Valid_MR_decoder': mr,
                       'Valid_MRR_decoder': mrr,
                       'Valid_Hits@1_decoder': hits_1,
                       'Valid_Hits@3_decoder': hits_3,
                       'Valid_Hits@10_decoder': hits_10,
                       'Train_Loss_decoder': loss_epoch})
        else:
            wandb.log({'Train_Loss_decoder': loss_epoch})


def load_decoder(args):
    decoder = get_decoder(args)
    model, _ = load_model(decoder, DECODER_FILE)
    return model


def evaluate_decoder(model, dataloader, fold, dev='cpu'):
    with torch.no_grad():
        model.eval()
        n, k = dataloader.get_properties()
        _, _, graph = dataloader.load(fold, dev)
        edge_idx_test, edge_type_test = dataloader.graph2idx(graph, dev)
        h, g = dataloader.load_embedding(dev)

        m = edge_idx_test.shape[1]
        ranks_head = []

        for i in range(m):
            triplet_idx = edge_idx_test[:, i]
            triplet_type = edge_type_test[i]

            triplets_head, position_head = dataloader.corrupt_triplet(n, triplet_idx)
            # triplets_tail, position_tail = dataloader.corrupt_triplet(n, triplet,head=False)

            triplets_head_type = torch.zeros(triplets_head.shape[1]).long()
            triplets_head_type[:] = triplet_type
            scores_head = model(h, g, triplets_head, triplets_head_type).cpu().numpy()

            rank_head = rank_triplet(scores_head, position_head)
            ranks_head.append(rank_head)
        ranks_head = np.array(ranks_head)

        return ranks_head


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=10, help="After how many epochs to evaluate.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs for decoder.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay decoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training")
    parser.add_argument("--batch_size", type=int, default=64000, help="Batch size for decoder.")

    # objective function parameters
    parser.add_argument("--margin", type=int, default=1, help="Margin for loss function.")

    # decoder parameters
    parser.add_argument("--channels", type=int, default=50, help="Number of channels for decoder.")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    wandb.init(project="KBAT_decoder", config=args)

    dataset = FB15Dataset()
    data_loader = DataLoader(dataset)

    train_decoder(args, data_loader)

    decoder = load_decoder(args)
    ranks = evaluate_decoder(decoder, data_loader, 'test', dev=args.device)

    mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranks)

    wandb.log({'Test_MR_decoder': mr,
               'Test_MRR_decoder': mrr,
               'Test_Hits@1_decoder': hits_1,
               'Test_Hits@3_decoder': hits_3,
               'Test_Hits@10_decoder': hits_10})


if __name__ == "__main__":
    main()
