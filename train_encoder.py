import argparse
import time

import torch
import torch.optim as optim
import torch.nn.functional as F

import wandb

from data.dataset import FB15, WN18, KINSHIP
from dataloader import DataLoader
from metrics import get_model_metrics
from utilis import save_best_encoder, set_random_seed, save_model, ENCODER, KBAT, DKBAT, get_encoder, get_data_loader


def train_encoder(args, model, data_loader):
    model_name = args.model + "_encoder"
    # set up weights and biases
    if args.debug == 1:
        model_name += '_debug'

    wandb.init(project=model_name, config=args)

    wandb.watch(model, log="all")

    # system parameters
    dev = args.device
    eval = args.eval

    # training parameters
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    step_size = args.step_size
    batch_size = args.batch
    negative_ratio = args.negative_ratio

    # encoder save file path
    dataset_name = data_loader.get_name()
    encoder_file = ENCODER + '_' + args.model.lower() + '_' + dataset_name.lower() + '.pt'

    # load data
    x, g, graph = data_loader.load_train('cpu')
    n = x.shape[0]

    # load graph base structure
    edge_idx, edge_type = data_loader.graph2idx(graph, dev='cpu')
    # load train edges
    train_idx, train_type = edge_idx.clone(), edge_type.clone()
    # load invalid negative sampling for train set
    train_head_invalid_sampling, train_tail_invalid_sampling = data_loader.load_invalid_sampling(fold='train')

    # load validation set for model metric
    _, _, valid_graph = data_loader.load_valid('cpu')
    valid_idx, valid_type = data_loader.graph2idx(valid_graph)
    # load invalid negative sampling for valid set
    valid_head_invalid_sampling, valid_tail_invalid_sampling = data_loader.load_invalid_sampling('valid')
    valid_pos_edge_epoch_idx, valid_neg_edge_idx, valid_pos_edge_epoch_type = data_loader.negative_samples(n, valid_idx,
                                                                                                           valid_type,
                                                                                                           negative_ratio,
                                                                                                           valid_head_invalid_sampling,
                                                                                                           valid_tail_invalid_sampling,
                                                                                                           'cpu')

    # optimizer and scheduler for training
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, last_epoch=-1, gamma=0.5)

    for epoch in range(epochs):
        s_epoch = time.time()
        model.train()

        # shuffling epoch training data
        perm = torch.randperm(train_idx.shape[1])
        train_idx = train_idx[:, perm]
        train_type = train_type[perm]
        train_head_invalid_sampling = train_head_invalid_sampling[perm]
        train_tail_invalid_sampling = train_tail_invalid_sampling[perm]

        losses_epoch = []
        m = train_idx.shape[1]  # all train edges
        for itt in range(0, m, batch_size):
            s_batch = time.time()
            # get batch boundaries
            start = itt
            end = itt + batch_size

            # get batch data
            batch_idx = train_idx[:, start:end]
            batch_type = train_type[start:end]
            batch_head_invalid_sampling = train_head_invalid_sampling[start:end]
            batch_tail_invalid_sampling = train_tail_invalid_sampling[start:end]

            # get positive and negative samples for batch
            s_sampling = time.time()
            batch_pos_idx, batch_neg_idx, batch_type = data_loader.negative_samples(n, batch_idx, batch_type,
                                                                                    negative_ratio,
                                                                                    batch_head_invalid_sampling,
                                                                                    batch_tail_invalid_sampling,
                                                                                    'cpu')
            t_sampling = time.time()

            # forward pass the model; getting the node embeddings out of the structural information
            s_forward = time.time()
            h_prime, g_prime = model(edge_idx.to(dev), edge_type.to(dev))
            t_forward = time.time()

            # compute model loss for positive and negative samples
            s_loss = time.time()
            loss = model.loss(h_prime, g_prime,
                              batch_pos_idx.to(dev),
                              batch_type.to(dev),
                              batch_neg_idx.to(dev),
                              batch_type.to(dev))
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
                print('Batch time: %.2f ' % (t_batch - s_batch) +
                      'Sampling time: %.4f ' % (t_sampling - s_sampling) +
                      'Forward time: %.2f ' % (t_forward - s_forward) +
                      'Loss time: %.2f ' % (t_loss - s_loss) +
                      'Optim time: %.2f ' % (t_optim - s_optim) +
                      'Loss Batch: %.4f ' % (losses_epoch[-1]))

        loss_epoch = sum(losses_epoch) / len(losses_epoch)

        # compute validation loss
        model.eval()
        h_prime, g_prime = model(train_idx.to(dev), train_type.to(dev))
        valid_loss = model.loss(h_prime, g_prime, valid_pos_edge_epoch_idx.to(dev), valid_pos_edge_epoch_type.to(dev),
                                valid_neg_edge_idx.to(dev), valid_pos_edge_epoch_type.to(dev)).item()

        scheduler.step()

        t_epcoh = time.time()
        if args.debug == 1:
            print('Epoch time: %.4f ' % (t_epcoh - s_epoch) +
                  'Loss Epoch: %.4f ' % loss_epoch +
                  'Loss Valid: %.4f ' % valid_loss)

        save_best_encoder(model, args.model, h_prime, g_prime, loss_epoch, epoch + 1, encoder_file, args, asc=False)
        torch.cuda.empty_cache()
        if (epoch + 1) % eval == 0:
            encoder_epoch_file = ENCODER + '_' + args.model.lower() + '_' + dataset_name.lower() + '_' + str(
                epoch) + '.pt'
            save_model(model, loss_epoch, epoch + 1, encoder_epoch_file, args)
            metrics = get_model_metrics(data_loader, h_prime, g_prime, 'valid', model, ENCODER, dev=args.device)
            metrics['train_' + dataset_name + '_Loss_encoder'] = loss_epoch
            metrics['valid_' + dataset_name + '_Loss_encoder'] = valid_loss
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_encoder': loss_epoch,
                       'valid_' + dataset_name + '_Loss_encoder': valid_loss})

        del h_prime, g_prime, loss
        torch.cuda.empty_cache()

    del x, g, graph, model, valid_loss
    torch.cuda.empty_cache()


def embed_nodes(args, encoder, data):
    dev = args.device

    data_loader = DataLoader(data)
    x, g, graph = data_loader.load_train(dev)
    edge_idx, edge_type = data_loader.graph2idx(graph, dev=dev)

    encoder.eval()
    with torch.no_grad():
        h, g = encoder(x, g, edge_idx, edge_type)
    return h, g


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=1, help="After how many epochs to evaluate.")
    parser.add_argument("--debug", type=int, default=1, help="Debugging mod.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for encoder.")
    parser.add_argument("--step_size", type=int, default=500, help="Step size of scheduler.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay encoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training.")
    parser.add_argument("--dataset", type=str, default=KINSHIP, help="Dataset used for training.")
    parser.add_argument("--batch", type=int, default=272115, help="Batch size.")
    parser.add_argument("--negative_ratio", type=int, default=2, help="Number of negative edges per positive one.")

    # objective function parameters
    parser.add_argument("--margin", type=int, default=5, help="Margin for loss function.")

    # encoder parameters
    parser.add_argument("--negative_slope", type=float, default=0.2, help="Negative slope for Leaky Relu")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads per layer")
    parser.add_argument("--hidden_encoder", type=int, default=200, help="Number of neurons per hidden layer")
    parser.add_argument("--output_encoder", type=int, default=400, help="Number of neurons per output layer")
    parser.add_argument("--model", type=str, default=KBAT, help='Model name')

    args, cmdline_args = parser.parse_known_args()

    data_loader = get_data_loader(args.dataset)

    # load model architecture
    x, g, _ = data_loader.load('train')
    model = get_encoder(args, x, g)

    # train model and save embeddings
    train_encoder(args, model, data_loader)


if __name__ == "__main__":
    main()
