import argparse
import time

import torch
import torch.optim as optim
import wandb

from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import get_model_metrics
from model import KBNet, DKBATNet
from utilis import save_best, load_model, set_random_seed

ENCODER = 'encoder'
DKBAT = 'DKBAT'
KBAT = 'KBAT'
ENCODER_FILE = 'encoder.pt'
ENCODER_CHECKPOINT = 'encoder_check.pt'


def get_encoder(args, x_size, g_size):
    # model parameters
    model_name = args.model
    h_size = args.hidden_encoder
    o_size = args.output_encoder
    heads = args.heads
    margin = args.margin
    alpha = args.alpha
    dropout = args.dropout
    negative_slope = args.negative_slope

    dev = args.device

    model = None
    if model_name == KBAT:
        model = KBNet(x_size, g_size, h_size, o_size, heads, margin, dropout, negative_slope=negative_slope, device=dev)
    elif model_name == DKBAT:
        model = DKBATNet(x_size, g_size, h_size, o_size, heads, alpha, margin, dropout, negative_slope=negative_slope,
                         device=dev)

    return model


def train_encoder(args, model, data_loader):
    # system parameters
    dev = args.device
    eval = args.eval

    # training parameters
    negative_ratio = 2
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    step_size = args.step_size

    dataset_name = data_loader.get_name()
    encoder_file = ENCODER + '_' + args.model.lower() + '_' + dataset_name.lower() + '.pt'

    # load data
    x, g, graph = data_loader.load_train(dev)
    wandb.watch(model, log="all")

    first = 0

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5, last_epoch=-1)

    train_idx, train_type = data_loader.graph2idx(graph, dev='cpu')
    n = x.shape[0]

    pos_edge_idx, pos_edge_type = train_idx[:, :], train_type[:]

    pos_edge_idx_aux = pos_edge_idx.repeat((1, negative_ratio))
    pos_edge_type_aux = pos_edge_type.repeat((1, negative_ratio))

    batch_size = train_idx.shape[1] * 10  # for cluster * 5
    for epoch in range(first, epochs):
        s_epoch = time.time()
        model.train()

        # negative sampling
        s_sampling = time.time()
        neg_edge_idx, neg_edge_type = data_loader.negative_samples(n, pos_edge_idx, pos_edge_type, negative_ratio,
                                                                   'cpu')
        t_sampling = time.time()

        # shuffling
        s_shuffling = time.time()
        perm = torch.randperm(pos_edge_type_aux.shape[1])
        pos_edge_idx_aux = pos_edge_idx_aux[:, perm]
        pos_edge_type_aux = pos_edge_type_aux[:, perm]
        t_shuffling = time.time()

        m = pos_edge_idx_aux.shape[1]
        iterations = torch.tensor([i for i in range(m)]).long()

        losses_epoch = []

        for itt in range(0, m, batch_size):
            s_batch = time.time()
            batch = iterations[itt:itt + batch_size]

            s_forward = time.time()
            h_prime, g_prime = model(x, g, train_idx.to(dev), train_type.to(dev))
            t_forward = time.time()

            s_slicing = time.time()
            pos_edge_idx_batch = pos_edge_idx_aux[:, batch].to(dev)
            pos_edge_type_batch = pos_edge_type_aux[:, batch].to(dev)
            neg_edge_idx_batch = neg_edge_idx[:, batch].to(dev)
            neg_edge_type_batch = neg_edge_type[:, batch].to(dev)
            t_slicing = time.time()

            s_loss = time.time()
            loss = model.loss(h_prime, g_prime,
                              pos_edge_idx_batch,
                              pos_edge_type_batch,
                              neg_edge_idx_batch,
                              neg_edge_type_batch)
            t_loss = time.time()

            torch.cuda.empty_cache()

            # optimization
            s_optim = time.time()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t_optim = time.time()

            losses_epoch.append(loss.item())

            t_batch = time.time()

            if args.debug == 1:
                wandb.log({'Batch time: %.2f': (t_batch - s_batch),
                           'Forward time: %.2f': (t_forward - s_forward),
                           'Slicing time: %.2f': (t_slicing - s_slicing),
                           'Loss time: %.2f': (t_loss - s_loss),
                           'Optim time: %.2f': (t_optim - s_optim)})
        loss_epoch = sum(losses_epoch) / len(losses_epoch)

        t_epcoh = time.time()
        if args.debug == 1:
            wandb.log({'Epoch time: %.4f': (t_epcoh - s_epoch),
                       'Sampling time: %.4f': (t_sampling - s_sampling),
                       'Shuffling time: %.4f': (t_shuffling - s_shuffling)})

        scheduler.step()
        save_best(model, loss_epoch, epoch + 1, encoder_file, asc=False)
        torch.cuda.empty_cache()
        if (epoch + 1) % eval == 0:
            model.eval()
            h_prime, g_prime = model(x, g, train_idx.to(dev), train_type.to(dev))
            metrics = get_model_metrics(data_loader, h_prime, g_prime, 'valid', model, ENCODER, dev=args.device)
            metrics['train_' + dataset_name + '_Loss_encoder'] = loss_epoch
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_encoder': loss_epoch})

        del h_prime, g_prime, loss
        torch.cuda.empty_cache()

    del pos_edge_idx, pos_edge_type, x, g, graph, model
    torch.cuda.empty_cache()


def embed_nodes(args, encoder, data):
    dev = args.device

    data_loader = DataLoader(data)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, paths=True, dev=dev)

    encoder.eval()
    with torch.no_grad():
        h, g = encoder(x, g, edge_idx, edge_type)
    return h, g


def load_encoder(args, g_size, x_size):
    encoder = get_encoder(args, x_size, g_size)
    encoder_file = ENCODER_FILE + '_' + args.model.lower() + '_' + args.dataset.lower() + '.pt'
    encoder, _ = load_model(encoder, encoder_file)
    return encoder


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=3000, help="After how many epochs to evaluate.")
    parser.add_argument("--debug", type=int, default=0, help="Debugging mod.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs for encoder.")
    parser.add_argument("--step_size", type=int, default=500, help="Step size of scheduler.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
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
    parser.add_argument("--alpha", type=float, default=0.5, help="Inbound neighborhood importance.")
    parser.add_argument("--model", type=str, default=KBAT, help='Model name')

    args, cmdline_args = parser.parse_known_args()

    model_name = args.model + "_encoder"
    # set up weights and biases
    if args.debug == 1:
        model_name += '_debug'
    wandb.init(project=model_name, config=args)

    if args.dataset == 'FB15k-237':
        dataset = FB15Dataset()
    else:
        dataset = WN18RR()
    data_loader = DataLoader(dataset)

    # load model architecture
    x_size = dataset.size_x
    g_size = dataset.size_g
    model = get_encoder(args, x_size, g_size)

    # train model and save embeddings
    train_encoder(args, model, data_loader)
    h, g = embed_nodes(args, model, dataset)
    dataset.save_embedding(h, g, args.model)


if __name__ == "__main__":
    main()
