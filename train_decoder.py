import argparse
import concurrent.futures

import torch
import numpy as np
import wandb

from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import get_model_metrics
from model import ConvKB
from utilis import load_model, save_model, save_best, set_random_seed

DECODER = 'decoder'

DECODER = 'decoder'

DECODER_FILE = 'decoder.pt'
DECODER_CHECKPOINT = 'decoder_checkpoint.pt'


def get_decoder(args):
    channels = args.channels
    dropout = args.dropout
    input_size = args.output_encoder
    dev = args.device

    model = ConvKB(input_dim=input_size, input_seq_len=3, in_channels=1, out_channels=channels, drop_prob=dropout,
                   dev=dev)
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
    model = args.model

    _, _, graph = data_loader.load_train('cpu')
    h, g = data_loader.load_embedding(model, 'cpu')

    decoder_file = DECODER_FILE + '_' + model.lower() + '_' + dataset_name.lower() + '.pt'

    first = 0

    optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=25, gamma=0.5, last_epoch=-1)

    pos_edge_idx, pos_edge_type = data_loader.graph2idx(graph, paths=False, dev='cpu')
    m = pos_edge_idx.shape[1]
    n = h.shape[0]

    criterion = torch.nn.SoftMarginLoss()
    for epoch in range(first, epochs):
        decoder.train()

        # defining an iteration
        iteration = torch.randperm(m).to(dev)
        loss_epoch = 0
        no_batch = int(m / batch_size)
        batch_counter = 0

        for itt in range(0, m, batch_size):
            batch = iteration[itt:itt + batch_size]
            pos_batch_idx = pos_edge_idx[:, batch]
            pos_batch_type = pos_edge_type[batch]

            # generate invalid batch triplets
            neg_batch_idx, neg_batch_type = data_loader.negative_samples(n, pos_batch_idx, pos_batch_type,
                                                                         negative_ratio,
                                                                         'cpu')

            # combine positive and negative batch
            no_pos = pos_batch_idx.shape[1]
            no_neg = neg_batch_idx.shape[1]

            target_batch = torch.cat([torch.ones(no_pos), -torch.ones(no_neg)])
            batch_idx = torch.cat([pos_batch_idx, neg_batch_idx], dim=1)
            batch_type = torch.cat([pos_batch_type, neg_batch_type])
            row, col = batch_idx

            # prepare input
            conv_input = torch.stack([h[row], g[batch_type], h[col]], dim=1).to(dev)

            # forward input
            prediction = decoder(conv_input)

            # compute loss
            loss = criterion(prediction.squeeze(-1), target_batch.to(dev))

            # optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_epoch += loss.item() / no_batch
            batch_counter += 1
            print('Epoch:%3.d ' % epoch +
                  'Iteration:%3.d ' % batch_counter +
                  'Loss iteration:%.4f ' % loss.item())
            del prediction, loss
            torch.cuda.empty_cache()
        scheduler.step()

        # save_model(decoder, loss_epoch, epoch + 1, DECODER_CHECKPOINT)
        save_best(decoder, loss_epoch, epoch + 1, decoder_file, asc=False)

        if (epoch + 1) % eval == 0:
            metrics = get_model_metrics(decoder, h, g, data_loader, 'valid', DECODER, dev)

            metrics['train_' + dataset_name + '_Loss_decoder'] = loss_epoch
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_decoder': loss_epoch})


def load_decoder(args):
    decoder = get_decoder(args)
    decoder_file = DECODER_FILE + '_' + args.model.lower() + '_' + args.dataset.lower() + '.pt'
    model, _ = load_model(decoder, decoder_file)
    return model


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=200, help="After how many epochs to evaluate.")
    parser.add_argument("--debug", type=int, default=0, help="Debugging mod.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs for decoder.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay decoder.")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for decoder.")
    parser.add_argument("--negative-ratio", type=int, default=40, help="Number of negative samples.")
    parser.add_argument("--dataset", type=str, default='FB15k-237', help="Dataset used for training.")
    parser.add_argument("--model", type=str, default='KBAT', help="Which model's embedding to use.")
    # objective function parameters
    parser.add_argument("--margin", type=int, default=1, help="Margin for loss function.")

    # decoder parameters
    parser.add_argument("--channels", type=int, default=50, help="Number of channels for decoder.")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    model_name = args.model + "_ConvKB"
    if args.debug:
        model_name += '_debug'
    wandb.init(project=model_name, config=args)

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
    print('done training!')

    h, g = data_loader.load_embedding(args.model)
    # Evaluate test and valid fold after training is done
    metrics = get_model_metrics(decoder, h, g, data_loader, 'test', DECODER, args.device)
    print('done eval test!')
    wandb.log(metrics)


if __name__ == "__main__":
    main()
