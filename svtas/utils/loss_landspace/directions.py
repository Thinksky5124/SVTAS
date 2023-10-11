'''
Author       : Thyssen Wen
Date         : 2023-04-07 15:11:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 15:12:39
Description  : file content
FilePath     : /SVTAS/svtas/utils/loss_landspace/directions.py
'''
import torch
from svtas.utils import is_h5py_available
if is_h5py_available():
    import h5py
from . import h5_utils


def create_random_directions(model, plot_1D=False):
    x_direction = create_random_direction(model)
    if not plot_1D:
        y_direction = create_random_direction(model)
        return [x_direction, y_direction]
    else:
        return [x_direction]


def create_random_direction(model):
    weights = get_weights(model)
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights)

    return direction


def get_weights(model):
    return [p.data for p in model.parameters()]


def get_random_weights(weights):
    return [torch.randn(w.size()) for w in weights]


def normalize_direction(direction, weights):
    for d, w in zip(direction, weights):
        d.mul_(w.norm() / (d.norm() + 1e-10))


def normalize_directions_for_weights(direction, weights):
    assert (len(direction) == len(weights))
    for d, w in zip(direction, weights):
        normalize_direction(d, w)

def get_diff_weights(weights, weights2):
    """ Produce a direction from 'weights' to 'weights2'."""
    return [w2 - w for (w, w2) in zip(weights, weights2)]

def get_diff_states(states, states2):
    """ Produce a direction from 'states' to 'states2'."""
    return [v2 - v for (k, v), (k2, v2) in zip(states.items(), states2.items())]

def ignore_biasbn(directions):
    """ Set bias and bn parameters in directions to zero """
    for d in directions:
        if d.dim() <= 1:
            d.fill_(0)

def load_directions(dir_file):
    """ Load direction(s) from the direction file."""

    f = h5py.File(dir_file, 'r')
    if 'ydirection' in f.keys():  # If this is a 2D plot
        xdirection = h5_utils.read_list(f, 'xdirection')
        ydirection = h5_utils.read_list(f, 'ydirection')
        directions = [xdirection, ydirection]
    else:
        directions = [h5_utils.read_list(f, 'xdirection')]

    return directions