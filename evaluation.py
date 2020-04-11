import argparse

import torch
import wandb
from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import get_model_metrics, get_model_metrics_head_or_tail
from train_decoder import get_decoder
from train_encoder import KBAT, DKBAT, get_encoder
from utilis import load_decoder_eval, load_encoder_eval

RUN_DIR = './eval_dir'


def evaluate_encoder(data_loader, fold, encoder, embedding_model, head, dev='cpu'):
    x, g, graph = data_loader.load_train()
    edge_idx, edge_idx = DataLoader.graph2idx(graph, paths=True, dev=dev)
    with torch.no_grad():
        encoder.eval()
        h, g = encoder(edge_idx, edge_idx)

    metrics = get_model_metrics_head_or_tail(data_loader, h, g, fold, encoder, 'encoder', head, dev=dev)
    print(embedding_model + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def evaluate_decoder(data_loader, fold, decoder, run_dir, model_name, head, dev='cpu'):
    h, g = data_loader.load_embedding(model_name)
    dataset_name = data_loader.get_name()
    load_decoder_eval(decoder, run_dir, model_name, dataset_name)

    metrics = get_model_metrics_head_or_tail(data_loader, h, g, fold, decoder, 'decoder', head, dev=dev)
    print(model_name + '_ConvKB ' + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def main_encoder():
    parser = argparse.ArgumentParser()

    parser.add_argument("--negative_slope", type=float, default=0.2, help="Negative slope for Leaky Relu")
    parser.add_argument("--heads", type=int, default=2, help="Number of heads per layer")
    parser.add_argument("--hidden_encoder", type=int, default=200, help="Number of neurons per hidden layer")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")
    parser.add_argument("--alpha", type=float, default=0.5, help="Inbound neighborhood importance.")
    parser.add_argument("--model", type=str, default=DKBAT, help='Model name')

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training.")
    parser.add_argument("--margin", type=int, default=1, help="Margin for loss function.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")

    # evaluation parameters
    parser.add_argument("--dataset", type=str, default='FB15k-237', help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='test', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")

    args, cmdline_args = parser.parse_known_args()

    model_name = args.model + "_encoder"
    head = True if args.head == 1 else 0
    prefix = 'head' if head else 'tail'

    fold = args.fold
    if args.dataset == 'FB15k-237':
        dataset = FB15Dataset()
    else:
        dataset = WN18RR()
    data_loader = DataLoader(dataset)

    wandb.init(project=model_name + '_' + dataset.name + '_' + fold + '_' + prefix + '_eval', config=args)

    # load model architecture
    x_size = dataset.size_x
    g_size = dataset.size_g
    model = get_encoder(args, x_size, g_size)

    dataset_name = data_loader.get_name()
    model = load_encoder_eval(model, RUN_DIR, model_name, dataset_name)

    fold = args.fold
    head = args.head
    evaluate_encoder(data_loader, fold, model, args.model, head, dev=args.device)


def main_decoder():
    parser = argparse.ArgumentParser()

    parser.add_argument("--channels", type=int, default=50, help="Number of channels for decoder.")
    parser.add_argument("--output_encoder", type=int, default=200, help="Number of neurons per output layer")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout for training")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--model", type=str, default=DKBAT, help='Model name')

    # evaluation parameters
    parser.add_argument("--dataset", type=str, default='WN18RR', help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='test', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")

    args, cmdline_args = parser.parse_known_args()

    decoder = get_decoder(args)

    if args.dataset == 'FB15k-237':
        dataset = FB15Dataset()
    else:
        dataset = WN18RR()
    data_loader = DataLoader(dataset)

    head = True if args.head == 1 else 0
    prefix = 'head' if head else 'tail'

    fold = args.fold
    model_name = args.model

    wandb.init(project=args.model + '_' + dataset.name + '_' + fold + '_' + prefix + '_eval', config=args)

    evaluate_decoder(data_loader, fold, decoder, RUN_DIR, model_name, head, dev=args.device)


if __name__ == '__main__':
    main_encoder()
