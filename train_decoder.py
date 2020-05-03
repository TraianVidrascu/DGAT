import argparse
import time

import torch
import torch.nn as nn
import wandb
from torch.autograd import Variable

from data.dataset import FB15, WN18, KINSHIP
from metrics import get_model_metrics
from utilis import save_best_decoder, set_random_seed, load_embedding, save_embeddings, \
    DECODER, EMBEDDING_DIR, KBAT, get_decoder, DECODER_NAME, get_data_loader, save_model, DKBAT


def train_decoder(args, decoder, data_loader, h, g):
    wandb.watch(decoder, log="all")

    dataset_name = data_loader.get_name()

    dev = args.device
    negative_ratio = args.negative_ratio
    lr = args.lr
    decay = args.decay
    epochs = args.epochs
    eval = args.eval
    batch_size = args.batch_size
    step_size = args.step_size
    model = args.model

    _, _, graph = data_loader.load_train('cpu')

    decoder_file = DECODER_NAME + '_' + model.lower() + '_' + dataset_name.lower() + '.pt'

    first = 0

    optim = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=step_size, gamma=0.5, last_epoch=-1)

    train_pos_idx, train_pos_type = data_loader.graph2idx(graph, dev='cpu')
    train_head_invalid_sampling, train_tail_invalid_sampling = data_loader.load_invalid_sampling('train')

    m = train_pos_idx.shape[1]
    n = h.shape[0]

    criterion = nn.SoftMarginLoss()
    for epoch in range(first, epochs):
        decoder.train()

        # data shuffling
        perm = torch.randperm(m)
        train_pos_idx = train_pos_idx[:, perm]
        train_pos_type = train_pos_type[perm]
        train_head_invalid_sampling = train_head_invalid_sampling[perm]
        train_tail_invalid_sampling = train_tail_invalid_sampling[perm]

        losses = []
        batch_counter = 0
        s_epoch = time.time()
        for itt in range(0, m, batch_size):
            s_batch = time.time()
            # get batch boundaries
            start = itt
            end = itt + batch_size

            batch_pos_idx = train_pos_idx[:, start:end]
            batch_pos_type = train_pos_type[start:end]
            batch_invalid_head_sampling = train_head_invalid_sampling[start:end]
            batch_tail_invalid_sampling = train_tail_invalid_sampling[start:end]

            s_sampling = time.time()
            # generate invalid batch triplets
            _, batch_neg_idx, batch_neg_type = data_loader.negative_samples(n, batch_pos_idx, batch_pos_type,
                                                                            negative_ratio, batch_invalid_head_sampling,
                                                                            batch_tail_invalid_sampling,
                                                                            'cpu')
            t_sampling = time.time()
            # combine positive and negative batch
            no_pos = batch_pos_idx.shape[1]
            no_neg = batch_neg_idx.shape[1]

            target_batch = Variable(torch.cat([torch.ones(no_pos), -torch.ones(no_neg)]))
            batch_idx = Variable(torch.cat([batch_pos_idx, batch_neg_idx], dim=1))
            batch_type = Variable(torch.cat([batch_pos_type, batch_neg_type]))

            # forward input
            s_pred = time.time()
            prediction = decoder(batch_idx, batch_type)
            t_pred = time.time()

            # compute loss
            loss = criterion(prediction.squeeze(-1), target_batch.to(dev))

            # optimization
            optim.zero_grad()
            loss.backward()
            optim.step()
            losses.append(loss.item())
            batch_counter += 1
            t_batch = time.time()
            print('Epoch:%3.d ' % epoch +
                  'Iteration:%3.d ' % batch_counter +
                  'Loss iteration:%.4f ' % loss.item() +
                  'Time sampling:%.4f ' % (t_sampling - s_sampling) +
                  'Time prediction:%.4f ' % (t_pred - s_pred) +
                  'Time batch:%.4f ' % (t_batch - s_batch))
            del prediction, loss
            torch.cuda.empty_cache()
        scheduler.step()
        t_epoch = time.time()

        loss_epoch = sum(losses) / len(losses)
        print('\nEpoch:{0} Average Loss:{1:.6f}\n'.format(epoch + 1, loss_epoch) +
              ' Total time:%.4f' % (t_epoch - s_epoch))
        save_best_decoder(decoder, loss_epoch, epoch + 1, decoder_file, args, asc=False)

        if (epoch + 1) % eval == 0:
            decoder_epoch_file = DECODER_NAME + '_' + model.lower() + '_' + dataset_name.lower() + '_' + str(
                epoch + 1) + '.pt'
            save_model(decoder, loss_epoch, epoch + 1, decoder_epoch_file, args)
            metrics = get_model_metrics(data_loader, decoder.node_embeddings, decoder.rel_embeddings, 'test', decoder,
                                        DECODER, dev)
            metrics['train_' + dataset_name + '_Loss_decoder'] = loss_epoch
            wandb.log(metrics)
        else:
            wandb.log({'train_' + dataset_name + '_Loss_decoder': loss_epoch})


def main():
    set_random_seed()
    torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()

    # system parameters
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--eval", type=int, default=100, help="After how many epochs to evaluate.")
    parser.add_argument("--debug", type=int, default=0, help="Debugging mod.")

    # training parameters
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs for decoder.")
    parser.add_argument("--step_size", type=int, default=25, help="Step size of scheduler.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--decay", type=float, default=1e-5, help="L2 normalization weight decay decoder.")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout for training")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for decoder.")
    parser.add_argument("--negative-ratio", type=int, default=40, help="Number of negative samples.")
    parser.add_argument("--dataset", type=str, default=WN18, help="Dataset used for training.")
    parser.add_argument("--model", type=str, default=KINSHIP, help="Which model's embedding to use.")
    # objective function parameters

    # decoder parameters
    parser.add_argument("--channels", type=int, default=500, help="Number of channels for decoder.")

    args, cmdline_args = parser.parse_known_args()

    # set up weights adn biases
    model_name = args.model + "_ConvKB"
    if args.debug:
        model_name += '_debug'
    wandb.init(project=model_name, config=args)

    # load dataset
    data_loader = get_data_loader(args.dataset)

    # load model architecture

    h, g = load_embedding(args.model, EMBEDDING_DIR, args.dataset)
    decoder = get_decoder(args, h, g)
    save_embeddings(h, g, model_name)
    # train decoder model
    train_decoder(args, decoder, data_loader, h, g)
    print('done training!')


if __name__ == "__main__":
    main()
