import argparse
import wandb
from data.dataset import FB15Dataset, WN18RR
from dataloader import DataLoader
from metrics import get_model_metrics, get_model_metrics_head_or_tail
from train_decoder import get_decoder
from train_encoder import KBAT, DKBAT
from utilis import load_decoder_eval


def evaluate_encoder(data_loader, fold, encoder, embedding_model, head, dev='cpu'):
    h, g = data_loader.load_embedding(embedding_model)
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
    run_dir = './eval_dir'

    wandb.init(project=args.model + '_' + dataset.name + '_' + fold + '_' + prefix + '_eval', config=args)

    evaluate_decoder(data_loader, fold, decoder, run_dir, model_name, head, dev=args.device)


if __name__ == '__main__':
    main_decoder()
