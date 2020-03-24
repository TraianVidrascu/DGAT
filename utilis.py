import torch
import os.path as osp
import wandb


def save_model(model, metric, epoch, file):
    path = osp.join(wandb.run.dir, file)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric}, path)


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
