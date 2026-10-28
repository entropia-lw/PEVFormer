import inspect
import torch

def load_checkpoint(path):
    options = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    return torch.load(path, **options)

def inverse_sigmoid(x, eps=1e-05):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)
