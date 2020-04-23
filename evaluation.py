import argparse

import torch
import wandb
from data.dataset import FB15, WN18, KINSHIP
from dataloader import DataLoader
from metrics import get_model_metrics_head_or_tail
from utilis import load_decoder_eval, load_encoder_eval, load_embedding, save_eval_model, KBAT, EMBEDDING_DIR, DKBAT, \
    get_data_loader

import torch.nn.functional as F
import os.path as osp


def get_embeddings(data_loader, dev, encoder):
    x, g, graph = data_loader.load_train(dev)
    # normalize input
    x = F.normalize(x, p=2, dim=1).detach()
    g = F.normalize(g, p=2, dim=1).detach()
    edge_idx, edge_type = DataLoader.graph2idx(graph, dev=dev)
    with torch.no_grad():
        encoder.eval()
        h_prime, g_prime = encoder(x, g, edge_idx, edge_type)
    return h_prime, g_prime


def evaluate_encoder(data_loader, fold, encoder, embedding_model, head, dev='cpu'):
    h_prime, g_prime = get_embeddings(data_loader, dev, encoder)
    metrics = get_model_metrics_head_or_tail(data_loader, h_prime, g_prime, fold, encoder, 'encoder', head, dev=dev)
    print(embedding_model + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def save_embedding(data_loader, encoder, embedding_model, dev='cpu'):
    h_prime, g_prime = get_embeddings(data_loader, dev, encoder)

    data_name = data_loader.get_name()
    h_file = 'h_' + embedding_model.lower() + '_' + data_name.lower() + '.pt'
    g_file = 'g_' + embedding_model.lower() + '_' + data_name.lower() + '.pt'

    h_path = osp.join(EMBEDDING_DIR, h_file)
    g_path = osp.join(EMBEDDING_DIR, g_file)

    torch.save(h_prime, h_path)
    torch.save(g_prime, g_path)


def evaluate_decoder(data_loader, fold, model_name, head, dev='cpu'):
    dataset_name = data_loader.get_name()
    h, g = load_embedding(model_name, EMBEDDING_DIR, dataset_name)

    decoder, _, _ = load_decoder_eval(model_name, dataset_name)

    metrics = get_model_metrics_head_or_tail(data_loader, h, g, fold, decoder, 'decoder', head, dev=dev)
    print(model_name + '_ConvKB ' + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def main_encoder():
    parser = argparse.ArgumentParser()

    # evaluation parameters
    parser.add_argument("--model", type=str, default=KBAT, help="Model used for evaluation")
    parser.add_argument("--dataset", type=str, default=KINSHIP, help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='valid', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")

    parser.add_argument("--save", type=int, default=0, help="Save node embedding.")
    parser.add_argument("--eval", type=int, default=1, help="Evaluate encoder.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run model.")

    args, cmdline_args = parser.parse_known_args()

    model_name = "encoder_" + args.model
    head = True if args.head == 1 else False
    prefix = 'head' if head else 'tail'

    save = True if args.save == 1 else False
    eval = True if args.eval == 1 else False

    data_loader = get_data_loader(args.dataset)

    dataset_name = data_loader.get_name()
    fold = args.fold

    model, epochs, args_original = load_encoder_eval(model_name, data_loader)

    if save:
        save_embedding(data_loader, model, args.model, dev=args.device)

    if eval:
        wandb.init(project=model_name + '_' + dataset_name + '_' + fold + '_' + prefix + '_eval', config=args)
        save_eval_model(model, model_name, dataset_name, args_original)
        evaluate_encoder(data_loader, fold, model, args.model, head, dev=args.device)


def main_decoder():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--model", type=str, default=KBAT, help='Model name')

    # evaluation parameters
    parser.add_argument("--dataset", type=str, default=WN18, help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='test', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")

    args, cmdline_args = parser.parse_known_args()

    data_loader = get_data_loader(args.dataset)
    dataset_name = data_loader.get_name()

    head = True if args.head == 1 else False
    prefix = 'head' if head else 'tail'

    fold = args.fold
    model_name = args.model
    wandb.init(project=args.model + '_' + 'ConvKB_' + dataset_name + '_' + fold + '_' + prefix + '_eval', config=args)

    evaluate_decoder(data_loader, fold, model_name, head, dev=args.device)


if __name__ == '__main__':
    main_encoder()
