import torch
import os.path as osp
import wandb

from data.dataset import FB15, FB15Dataset, WN18, WN18RR, KINSHIP, Kinship
from dataloader import DataLoader
from model import KBNet, DKBATNet, ConvKB, WrapperConvKB

ENCODER = 'encoder'
DKBAT = 'DKBAT'
KBAT = 'KBAT'

ENCODER_FILE = 'encoder.pt'
DECODER_FILE = 'decoder.pt'

ENCODER_DIR = './eval_dir/encoder'
DECODER_DIR = './eval_dir/decoder'
EMBEDDING_DIR = './eval_dir/embeddings'

DECODER = 'decoder'
DECODER_NAME = 'ConvKB'


def get_encoder(args, x, g):
    # model parameters
    model_name = args.model
    h_size = args.hidden_encoder
    o_size = args.output_encoder
    heads = args.heads
    margin = args.margin
    dropout = args.dropout
    negative_slope = args.negative_slope

    dev = args.device

    model = None
    if model_name == KBAT:
        model = KBNet(x, g, h_size, o_size, heads, margin, dropout, negative_slope=negative_slope, device=dev)
    elif model_name == DKBAT:
        model = DKBATNet(x, g, h_size, o_size, heads, margin, dropout, negative_slope=negative_slope,
                         device=dev)

    return model


def get_decoder(args, h, g):
    channels = args.channels
    dropout = args.dropout
    input_size = h.shape[1]
    dev = args.device

    model = WrapperConvKB(h, g, input_dim=input_size, input_seq_len=3, in_channels=1, out_channels=channels,
                          drop_prob=dropout,
                          dev=dev)
    return model


def save_model(model, metric, epoch, file, args):
    path = osp.join(wandb.run.dir, file)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric, 'args': args}, path)


def save_eval_model(model, model_name, dataset_name, args):
    path = osp.join(wandb.run.dir, model_name.lower() + '_' + dataset_name.lower() + '.pt')
    torch.save({'model_state_dict': model.state_dict(), 'args': args, 'dataset': dataset_name}, path)


def save_best_decoder(model, metric, epoch, file, args, asc=True):
    path = osp.join(wandb.run.dir, file)
    metric_old = None
    if osp.exists(path):
        model_dict = torch.load(path)
        metric_old = model_dict['metric']

    if metric_old is None:
        save_model(model, metric, epoch, file, args)
    elif asc and metric > metric_old:
        save_model(model, metric, epoch, file, args)
    elif metric < metric_old:
        save_model(model, metric, epoch, file, args)


def save_best_encoder(model, model_name, h, g, metric, epoch, file, args, asc=True):
    path = osp.join(wandb.run.dir, file)
    metric_old = None
    if osp.exists(path):
        model_dict = torch.load(path)
        metric_old = model_dict['metric']

    if metric_old is None:
        save_model(model, metric, epoch, file, args)
        save_embeddings(h, g, model_name)
    elif asc and metric > metric_old:
        save_model(model, metric, epoch, file, args)
        save_embeddings(h, g, model_name)
    elif metric < metric_old:
        save_model(model, metric, epoch, file, args)
        save_embeddings(h, g, model_name)


def load_model(path):
    model_dict = torch.load(path)
    state_dict = model_dict['model_state_dict']
    epoch = model_dict['epoch']
    args = model_dict['args']
    return state_dict, epoch, args


def load_decoder_eval(encoder_name, dataset_name):
    path = osp.join(DECODER_DIR, DECODER_NAME + '_' + encoder_name.lower() + '_' + dataset_name.lower() + '.pt')
    state_dict, epoch, args = load_model(path)
    decoder = get_decoder(args)
    decoder.load_state_dict(state_dict)
    return decoder, epoch, args


def load_embedding(model_name, rundir, dataset_name):
    path_h = osp.join(rundir, 'h_' + model_name.lower() + '_' + dataset_name.lower() + '.pt')
    path_g = osp.join(rundir, 'g_' + model_name.lower() + '_' + dataset_name.lower() + '.pt')
    h = torch.load(path_h)
    g = torch.load(path_g)
    return h, g


def save_embeddings(h, g, model_name):
    path_h = osp.join(wandb.run.dir, 'h_' + model_name.lower() + '.pt')
    path_g = osp.join(wandb.run.dir, 'g_' + model_name.lower() + '.pt')
    torch.save(h, path_h)
    torch.save(g, path_g)


def load_encoder_eval(model_name, data_loader):
    x_size, g_size = data_loader.get_embedding_size()
    dataset_name = data_loader.get_name()

    path = osp.join(ENCODER_DIR, model_name.lower() + '_' + dataset_name.lower() + '.pt')
    state_dict, epoch, args = load_model(path)

    encoder = get_encoder(args, x_size, g_size)
    encoder.load_state_dict(state_dict)
    return encoder, epoch, args


def get_data_loader(dataset_name):
    if dataset_name == FB15:
        dataset = FB15Dataset()
    elif dataset_name == WN18:
        dataset = WN18RR()
    elif dataset_name == KINSHIP:
        dataset = Kinship()
    else:
        raise Exception('Database not found!')
    data_loader = DataLoader(dataset)
    return data_loader


def set_random_seed():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
