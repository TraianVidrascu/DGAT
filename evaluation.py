import argparse

import torch
import wandb
from data.dataset import FB15, WN18, KINSHIP
from dataloader import DataLoader
from metrics import get_model_metrics_head_or_tail
from model import KB
from utilis import load_decoder_eval, load_encoder_eval, load_embedding, save_eval_model, KBAT, EMBEDDING_DIR, DKBAT, \
    get_data_loader

import torch.nn.functional as F
import os.path as osp


def get_embeddings(data_loader, dev, encoder, use_paths, use_partial):
    x, g, graph = data_loader.load_train(dev)
    # normalize input

    edge_idx, edge_type = DataLoader.graph2idx(graph, dev=dev)
    path_idx, path_type = data_loader.load_paths(use_paths, use_partial, dev=dev)
    with torch.no_grad():
        encoder.eval()
        h_prime, g_prime = encoder(edge_idx, edge_type, path_idx, path_type, use_paths)
    return h_prime, g_prime


def evaluate_encoder(data_loader, fold, encoder, embedding_model, head, use_paths, use_partial, dev='cpu'):
    _, _, graph = data_loader.load_train()
    edge_idx, edge_type = data_loader.graph2idx(graph, dev)
    path_idx, path_type = data_loader.load_paths(use_paths, use_partial, dev)

    with torch.no_grad():
        encoder.eval()
        h, g = encoder(edge_idx, edge_type, path_idx, path_type, use_paths)
    metrics = get_model_metrics_head_or_tail(data_loader, h, g, fold, encoder, 'encoder', head, dev=dev)
    print(embedding_model + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def save_embedding(data_loader, encoder, embedding_model, use_paths, use_partial, dev='cpu'):
    h_prime, g_prime = get_embeddings(data_loader, dev, encoder, use_paths, use_partial)

    data_name = data_loader.get_name()
    h_file = 'h_' + embedding_model.lower() + '_' + data_name.lower() + '.pt'
    g_file = 'g_' + embedding_model.lower() + '_' + data_name.lower() + '.pt'

    h_path = osp.join(EMBEDDING_DIR, h_file)
    g_path = osp.join(EMBEDDING_DIR, g_file)

    torch.save(h_prime, h_path)
    torch.save(g_prime, g_path)


def evaluate_embeddings(data_loader, fold, h, g, head, dev):
    encoder = KB(h, g)
    metrics = get_model_metrics_head_or_tail(data_loader, h, g, fold, encoder, 'encoder', head, dev=dev)
    print(metrics)


def run_embeddings():
    parser = argparse.ArgumentParser()

    # evaluation parameters
    parser.add_argument("--model", type=str, default=KBAT, help="Model used for evaluation")
    parser.add_argument("--dataset", type=str, default=KINSHIP, help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='test', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run model.")

    args, cmdline_args = parser.parse_known_args()

    data_loader = get_data_loader(args.dataset)

    dataset_name = data_loader.get_name()
    fold = args.fold
    h, g = load_embedding(args.model, EMBEDDING_DIR, dataset_name)
    evaluate_embeddings(data_loader, fold, h, g, args.head, dev=args.device)


def evaluate_decoder(data_loader, fold, model_name, head, dev='cpu'):
    dataset_name = data_loader.get_name()
    h, g = load_embedding(model_name, EMBEDDING_DIR, dataset_name)

    decoder, _, _ = load_decoder_eval(model_name, data_loader, h, g)

    metrics = get_model_metrics_head_or_tail(data_loader, decoder.node_embeddings, decoder.rel_embeddings, fold,
                                             decoder, 'decoder', head, dev=dev)
    print(model_name + '_ConvKB ' + data_loader.get_name() + ' ' + fold + ' metrics:')
    print(metrics)
    wandb.log(metrics)


def main_encoder():
    parser = argparse.ArgumentParser()

    # evaluation parameters
    parser.add_argument("--model", type=str, default=KBAT, help="Model used for evaluation")
    parser.add_argument("--dataset", type=str, default=FB15, help="Dataset used for evaluation.")
    parser.add_argument("--fold", type=str, default='test', help="Fold used for evaluation.")
    parser.add_argument("--head", type=int, default=0, help="Head or tail evaluation.")

    parser.add_argument("--save", type=int, default=1, help="Save node embedding.")
    parser.add_argument("--eval", type=int, default=0, help="Evaluate encoder.")
    parser.add_argument("--device", type=str, default='cuda', help="Device to run model.")

    args, cmdline_args = parser.parse_known_args()

    model_name = "encoder_" + args.model
    head = True if args.head == 1 else False
    prefix = 'head' if head else 'tail'

    save = True if args.save == 1 else False
    eval = True if args.eval == 1 else False

    try:
        use_paths = args.use_paths == 1
        use_partial = args.use_partial == 1
    except AttributeError:
        use_paths = False
        use_partial = False

    data_loader = get_data_loader(args.dataset)

    dataset_name = data_loader.get_name()
    fold = args.fold

    model, epochs, args_original = load_encoder_eval(model_name, data_loader)

    if save:
        save_embedding(data_loader, model, args.model, use_paths, use_partial, dev=args.device)

    if eval:
        wandb.init(project=model_name + '_' + dataset_name + '_' + fold + '_' + prefix + '_eval', config=args)
        save_eval_model(model, model_name, dataset_name, args_original)
        evaluate_encoder(data_loader, fold, model, args.model, head, use_paths, use_partial, dev=args.device)


def main_decoder():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default='cuda', help="Device to use for training.")
    parser.add_argument("--model", type=str, default=DKBAT, help='Model name')

    # evaluation parameters
    parser.add_argument("--dataset", type=str, default=FB15, help="Dataset used for evaluation.")
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
