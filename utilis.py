import torch
import os.path as osp
import wandb


def save_model(model, metric, epoch, file):
    path = osp.join(wandb.run.dir, file)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric}, path)


def save_eval_model(model, model_name, dataset_name):
    path = osp.join(wandb.run.dir, model_name.lower() + '_' + dataset_name.lower() + '.pt')
    torch.save({'model_state_dict': model.state_dict()}, path)


def save_best_decoder(model, metric, epoch, file, asc=True):
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


def save_best_encoder(model, model_name, h, g, metric, epoch, file, asc=True):
    path = osp.join(wandb.run.dir, file)
    metric_old = None
    if osp.exists(path):
        model_dict = torch.load(path)
        metric_old = model_dict['metric']

    if metric_old is None:
        save_model(model, metric, epoch, file)
        save_embeddings(h, g, model_name)
    elif asc and metric > metric_old:
        save_model(model, metric, epoch, file)
        save_embeddings(h, g, model_name)
    elif metric < metric_old:
        save_model(model, metric, epoch, file)
        save_embeddings(h, g, model_name)


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


def load_encoder_eval(model, rundir, model_name, dataset_name):
    path = osp.join(rundir, model_name.lower() + '_' + dataset_name.lower() + '.pt')
    model_dict = torch.load(path)['model_state_dict']
    model.load_state_dict(model_dict)
    return model


def set_random_seed():
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    torch.manual_seed(42)
