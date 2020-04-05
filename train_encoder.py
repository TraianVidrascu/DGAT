import argparse

import torch
import wandb

from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import get_metrics, evaluate
from model import KBNet, DKBATNet
from utilis import save_best, load_model, set_random_seed

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

    dev = args.device

    model = None
    if model_name == KBAT:
        model = KBNet(x_size, g_size, h_size, o_size, heads, margin, device=dev)
    elif model_name == DKBAT:
        model = DKBATNet(x_size, g_size, h_size, o_size, heads, alpha, margin, device=dev)

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
    use_paths = args.paths

    dataset_name = data_loader.get_name()

    # load data
    x, g, graph = data_loader.load_train(dev)
    wandb.watch(model, log="all")

    first = 0
    # if args.checkpoint:
    #     model, first = load_model(model, ENCODER_CHECKPOINT)

    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=500, gamma=0.5, last_epoch=-1)

    train_idx, train_type, pos_edge_idx, pos_edge_type = data_loader.graph2idx(graph, path=use_paths, dev='cpu')
    n = x.shape[0]

    pos_edge_idx_aux = pos_edge_idx.repeat((1, negative_ratio))
    pos_edge_type_aux = pos_edge_type.repeat((1, negative_ratio))

    batch_size = train_idx.shape[1] * 5  # for cluster

    for epoch in range(first, epochs):
        model.train()

        neg_edge_idx, neg_edge_type = data_loader.negative_samples(n, pos_edge_idx, pos_edge_type, negative_ratio,
                                                                   'cpu')

        # shuffling
        perm = torch.randperm(pos_edge_type_aux.shape[1])
        pos_edge_idx_aux = pos_edge_idx_aux[:, perm]
        pos_edge_type_aux = pos_edge_type_aux[:, perm]
        neg_edge_idx = neg_edge_idx[:, perm]
        neg_edge_type = neg_edge_type[:, perm]

        m = pos_edge_idx_aux.shape[1]
        iterations = torch.randperm(m)

        loss_epoch = 0
        no_batch = int(m / batch_size)

        for itt in range(0, m, batch_size):
            batch = iterations[itt:itt + batch_size]

            h_prime, g_prime = model(x, g, train_idx.to(dev), train_type.to(dev))

            pos_edge_idx_batch = pos_edge_idx_aux[:, batch].to(dev)
            pos_edge_type_batch = pos_edge_type_aux[:, batch].to(dev)
            neg_edge_idx_batch = neg_edge_idx[:, batch].to(dev)
            neg_edge_type_batch = neg_edge_type[:, batch].to(dev)

            loss = model.loss(h_prime, g_prime,
                              pos_edge_idx_batch,
                              pos_edge_type_batch,
                              neg_edge_idx_batch,
                              neg_edge_type_batch)

            torch.cuda.empty_cache()

            # optimization
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_epoch += loss.item() / no_batch
        scheduler.step()
        save_best(model, loss_epoch, epoch + 1, ENCODER_FILE, asc=False)
        torch.cuda.empty_cache()
        if (epoch + 1) % eval == 0:
            model.eval()
            h_prime, g_prime = model(x, g, train_idx.to(dev), train_type.to(dev))
            metrics = get_encoder_metrics(data_loader, h_prime, g_prime, 'valid', model, dev=args.device)
            metrics['train_' + dataset_name + '_Loss_encoder'] = loss_epoch
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_encoder': loss_epoch})

        del neg_edge_idx, h_prime, g_prime, loss
        torch.cuda.empty_cache()

    del pos_edge_idx, pos_edge_type, x, g, graph, model
    torch.cuda.empty_cache()


def get_ranking_metric(ranking_name, ranking, dataset_name, fold):
    mr, mrr, hits_1, hits_3, hits_10 = get_metrics(ranking)

    metrics = {fold + '_' + dataset_name + '_' + ranking_name + '_MR_encoder': mr,
               fold + '_' + dataset_name + '_' + ranking_name + '_MRR_encoder': mrr,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@1_encoder': hits_1,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@3_encoder': hits_3,
               fold + '_' + dataset_name + '_' + ranking_name + '_Hits@10_encoder': hits_10}
    return metrics


def get_encoder_metrics(data_loader, h, g, fold, encoder, dev='cpu'):
    ranks_head, ranks_tail, ranks = evaluate(encoder, h, g, data_loader, fold, dev)

    dataset_name = data_loader.get_name()

    metrics_head = get_ranking_metric('head', ranks_head, dataset_name, fold)
    metrics_tail = get_ranking_metric('tail', ranks_tail, dataset_name, fold)
    metrics_all = get_ranking_metric('both', ranks, dataset_name, fold)

    metrics = {**metrics_head, **metrics_tail, **metrics_all}
    return metrics


def embed_nodes(args, encoder, data):
    dev = args.device

    data_loader = DataLoader(data)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, path=False, dev=dev)

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
    parser.add_argument("--eval", type=int, default=500, help="After how many epochs to evaluate.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=3000, help="Number of training epochs for encoder.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay encoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training.")
    parser.add_argument("--dataset", type=str, default='FB15k-237', help="Dataset used for training.")
    parser.add_argument("--paths", type=bool, default=True, help="Use 2-hop paths for training.")

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

    model_name = args.model
    # set up weights and biases
    wandb.init(project=model_name + "_encoder", config=args)

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
    dataset.save_embedding(h, g)

    # evaluate test and valid fold after training
    metrics = get_encoder_metrics(data_loader, h, g, 'test', model, dev=args.device)
    wandb.log(metrics)


if __name__ == "__main__":
    main()
