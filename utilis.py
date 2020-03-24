import torch
import os.path as osp
import numpy as np

dir = './weights/'


def save_model(model, metric, epoch, file):
    path = osp.join(dir, file)
    torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch, 'metric': metric}, path)


def save_best(model, metric, epoch, file, asc=True):
    path = osp.join(dir, file)
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
    path = osp.join(dir, file)
    epoch = 0
    if osp.exists(path):
        model_dict = torch.load(path)
        model.load_state_dict(model_dict['model_state_dict'])
        epoch = model_dict['epoch']
    return model, epoch

def print_results(*list_args, **keyword_args):
    result_string = str()
    for key in keyword_args.keys():
        value = keyword_args[key]
        if type(value) is float or type(value) is np.float64:
            value = "{0:.4f}".format(value)

        line = key + ': ' + str(value) + '; '
        result_string += line

    print(result_string)