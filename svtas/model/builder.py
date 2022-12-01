'''
Author: Thyssen Wen
Date: 2022-04-14 16:16:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-30 19:16:36
Description: registry and builder model
FilePath     : /SVTAS/svtas/model/builder.py
'''
from ..utils.build import Registry
from ..utils.build import build

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
ARCHITECTURE = Registry('architecture')
LOSSES = Registry('loss')
POSTPRECESSING = Registry('post_precessing')

def build_backbone(cfg):
    """Build backbone."""
    if 'sbp_build' in cfg.keys():
        sbp_build = cfg.pop('sbp_build')
    else:
        sbp_build = False
    if 'keep_ratio' in cfg.keys():
        keep_ratio = cfg.pop('keep_ratio')
    else:
        keep_ratio = 0.125
    return build(cfg, BACKBONES, sbp_build=sbp_build, keep_ratio=keep_ratio)

def build_head(cfg):
    """Build head."""
    if 'sbp_build' in cfg.keys():
        sbp_build = cfg.pop('sbp_build')
    else:
        sbp_build = False
    if 'keep_ratio' in cfg.keys():
        keep_ratio = cfg.pop('keep_ratio')
    else:
        keep_ratio = 0.125
    return build(cfg, HEADS, sbp_build=sbp_build, keep_ratio=keep_ratio)

def build_neck(cfg):
    """Build neck."""
    if 'sbp_build' in cfg.keys():
        sbp_build = cfg.pop('sbp_build')
    else:
        sbp_build = False
    if 'keep_ratio' in cfg.keys():
        keep_ratio = cfg.pop('keep_ratio')
    else:
        keep_ratio = 0.125
    return build(cfg, NECKS, sbp_build=sbp_build, keep_ratio=keep_ratio)

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
