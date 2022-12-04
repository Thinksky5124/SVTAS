'''
Author: Thyssen Wen
Date: 2022-04-14 16:16:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-01 16:31:03
Description: registry and builder model
FilePath     : /SVTAS/svtas/model/builder.py
'''
from ..utils.build import Registry
from ..utils.build import build
from ..utils.sbp import StochasticBackpropagation

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
ARCHITECTURE = Registry('architecture')
LOSSES = Registry('loss')
POSTPRECESSING = Registry('post_precessing')

def build_backbone(cfg):
    """Build backbone."""
    kwargs=dict()
    for key in StochasticBackpropagation.SBP_ARGUMENTS:
        if key in cfg.keys():
            kwargs[key] = cfg.pop(key)
    return build(cfg, BACKBONES, **kwargs)

def build_head(cfg):
    """Build head."""
    kwargs=dict()
    for key in StochasticBackpropagation.SBP_ARGUMENTS:
        if key in cfg.keys():
            kwargs[key] = cfg.pop(key)
    return build(cfg, HEADS, **kwargs)

def build_neck(cfg):
    """Build neck."""
    kwargs=dict()
    for key in StochasticBackpropagation.SBP_ARGUMENTS:
        if key in cfg.keys():
            kwargs[key] = cfg.pop(key)
    return build(cfg, NECKS, **kwargs)

def build_post_precessing(cfg):
    """Build loss."""
    return build(cfg, POSTPRECESSING)

def build_loss(cfg):
    """Build loss."""
    return build(cfg, LOSSES)

def build_architecture(cfg):
    """Build recognizer."""
    return build(cfg, ARCHITECTURE, key='architecture')
    
def build_model(cfg):
    """Build model."""
    args = cfg.copy()
    obj_type = args.get('architecture')
    if obj_type in ARCHITECTURE:
        return build_architecture(cfg)
    raise ValueError(f'{obj_type} is not registered in '
                     'ARCHITECTURE')
