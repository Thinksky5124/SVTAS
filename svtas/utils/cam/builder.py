'''
Author       : Thyssen Wen
Date         : 2022-12-23 17:42:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-24 13:21:30
Description  : file content
FilePath     : /SVTAS/svtas/utils/cam/builder.py
'''
from . import target_utils as custom_model_targets
from pytorch_grad_cam.utils import model_targets
from . import match_fn

def get_model_target_class(target_name, cfg):
    target = getattr(model_targets, target_name, False)
    if target is False:
        target = getattr(custom_model_targets, target_name)(**cfg)
    else:
        target = target(**cfg)
    return target

def get_match_fn_class(fn_name):
    fn = getattr(match_fn, fn_name, False)
    return fn