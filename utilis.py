import torch
import os.path as osp
import wandb


def save_model(model, metric, epoch, file):
    path = osp.join(wandb.run.dir, file)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric}, path)
    wandb.save(path)


def save_best(model, metric, epoch, file, asc=True):
    path = osp.join(wandb.run.dir, file)
    metric_old = None
    if osp.exists(path):
        model_dict = torch.load(path)
        metric_old = model_dict['metric']

    if metric_old is None:
        save_model(model, metric, epoch, file)
    elif asc and metric > metric_old:
        save_model(model, metric, epoch, file)
    elif metric < metric_old:
        save_model(model, metric, epoch, file)


def load_model(model, file):
    path = osp.join(wandb.run.dir, file)
    epoch = 0
    if osp.exists(path):
        model_dict = torch.load(path)
        model.load_state_dict(model_dict['model_state_dict'])
        epoch = model_dict['epoch']
    return model, epoch


def load_decoder_eval(model, rundir, model_name, dataset_name):
    path = osp.join(rundir, 'decoder_' + model_name.lower() + '_' + dataset_name.lower() + '.pt')
    if osp.exists(path):
        model_dict = torch.load(path)
        model.load_state_dict(model_dict['model_state_dict'])
    return model


def load_encoder_eval(model, rundir, model_name, dataset_name):
    path = osp.join(rundir, model_name.lower() + '_' + dataset_name.lower() + '.pt')
    if osp.exists(path):
        model_dict = torch.load(path, map_location='cuda:0')
        model.load_state_dict(model_dict['model_state_dict'])
    return model


def set_random_seed():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
