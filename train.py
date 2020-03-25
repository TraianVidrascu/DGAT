import argparse

import torch
import numpy as np
import wandb

from data.dataset import FB15Dataset
from dataloader import DataLoader
from metrics import rank_triplet, get_metrics
from model import KBNet, ConvKB
from utilis import load_model, save_model, save_best

DECODER_FILE = 'decoder.pt'
DECODER_CHECKPOINT = 'decoder_checkpoint.pt'

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


def get_decoder(args):
    channels = args.channels
    dropout = args.dropout
    dev = args.device
    input_size = args.output_encoder

    model = ConvKB(input_size, channels, dropout=dropout, dev=dev)
    return model


def train_encoder(args, data_loader):
    # system parameters
    dev = args.device
    eval = args.eval_encoder
    # training parameters
    lr = args.lr
    decay = args.decay_encoder
    epochs = args.epochs_encoder

    # load data
    x, g, graph = data_loader.load_train(dev)

    # determine input size
    x_size = x.shape[1]
    g_size = g.shape[1]

    model = get_encoder(args, x_size, g_size)

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
            ranks = evaluate_encoder(h_prime, g_prime, data_loader, 'valid', model._dissimilarity,
                                     dev=dev)

            mr, mrr, hits = get_metrics(ranks)
            wandb.log({'Valid_MR_encoder': mr,
                       'Valid_MRR_encoder': mrr,
                       'Valid_Hits@10_encoder': hits,
                       'Train_Loss_encoder': loss})
        else:
            wandb.log({'Train_Loss_encoder': loss})

        del neg_edge_idx, h_prime, g_prime, loss
        torch.cuda.empty_cache()

    del pos_edge_idx, edge_type, x, g, graph, model
    torch.cuda.empty_cache()


def embed_nodes(args, data):
    dev = args.device

    data_loader = DataLoader(data)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, dev)

    # determine input size
    x_size = x.shape[1]
    g_size = g.shape[1]

    encoder = get_encoder(args, x_size, g_size)
    encoder, _ = load_model(encoder, ENCODER_FILE)

    encoder.eval()
    with torch.no_grad():
        h, g = encoder(x, g, edge_idx, edge_type)
    return h, g


def train_decoder(args, data_loader):
    dev = args.device
    lr = args.lr
    decay = args.decay_decoder
    epochs = args.epochs_decoder
    eval = args.eval_decoder
    batch_size = args.decoder_batch_size

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
            ranks = evaluate_decoder(decoder, data_loader, 'valid', max_batch=200, dev=args.device)

            mr, mrr, hits = get_metrics(ranks)
            wandb.log({'Valid_MR_decoder': mr,
                       'Valid_MRR_decoder': mrr,
                       'Valid_Hits@10_decoder': hits,
                       'Train_Loss_decoder': loss_epoch})
        else:
            wandb.log({'Train_Loss_decoder': loss_epoch})


def load_decoder(args):
    decoder = get_decoder(args)
    model, _ = load_model(decoder, DECODER_FILE)
    return model


def evaluate_encoder(h, g, dataloader, fold, score_fct, max_batch=100000, dev='cpu'):
    with torch.no_grad():
        n, k = dataloader.get_properties()
        _, _, graph = dataloader.load(fold, dev)
        edge_idx, edge_type = dataloader.graph2idx(graph, dev)
        m = edge_idx.shape[1]

        ranks_head = []

        m = min(max_batch, m)

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


def evaluate_decoder(model, dataloader, fold, max_batch=10000000, dev='cpu'):
    with torch.no_grad():
        model.eval()
        n, k = dataloader.get_properties()
        _, _, graph = dataloader.load(fold, dev)
        edge_idx_test, edge_type_test = dataloader.graph2idx(graph, dev)
        h, g = dataloader.load_embedding(dev)

        m = edge_idx_test.shape[1]
        ranks_head = []

        m = min(max_batch, m)

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


def set_random_seed():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval_encoder", type=int, default=100, help="After how many epochs to evaluate.")
    parser.add_argument("--eval_decoder", type=int, default=10, help="After how many epochs to evaluate.")
    parser.add_argument("--train_encoder", type=int, default=1, help="Train the encoder.")
    parser.add_argument("--train_decoder", type=int, default=1, help="Train the decoder.")
    # parser.add_argument("--checkpoint", type=bool, default=False, help="Use checkpoint.")

    # training parameters
    parser.add_argument("--epochs_encoder", type=int, default=3000, help="Number of training epochs for encoder.")
    parser.add_argument("--epochs_decoder", type=int, default=200, help="Number of training epochs for decoder.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument("--decay_encoder", type=float, default=1e-5, help="L2 normalization weight decay encoder.")
    parser.add_argument("--decay_decoder", type=float, default=1e-5, help="L2 normalization weight decay decoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training")
    parser.add_argument("--decoder_batch_size", type=int, default=16000, help="Batch size for decode.")

    # objective function parameters
    parser.add_argument("--margin", type=int, default=1e-5, help="Margin for loss function.")

    # encoder parameters
    parser.add_argument("--negative_slope", type=float, default=0.2, help="Negative slope for Leaky Relu")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads per layer")
    parser.add_argument("--hidden_encoder", type=int, default=200, help="Number of neurons per hidden layer")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")

    # decoder parameters
    parser.add_argument("--channels", type=int, default=50, help="Number of channels for decoder.")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    wandb.init(project="DGAT", config=args)

    dataset = FB15Dataset()
    data_loader = DataLoader(dataset)

    if args.train_encoder != 0:
        train_encoder(args, data_loader)
        h, g = embed_nodes(args, dataset)
        dataset.save_embedding(h, g)

    if args.train_decoder != 0:
        train_decoder(args, data_loader)

    decoder = load_decoder(args).to()
    ranks = evaluate_decoder(decoder, data_loader, 'test', dev=args.device)

    mr, mrr, hits = get_metrics(ranks)

    wandb.log({'Test_MR_decoder': mr,
               'Test_MRR_decoder': mrr,
               'Test_Hits@10_decoder': hits})


if __name__ == "__main__":
    main()
