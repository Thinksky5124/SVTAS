'''
Author       : Thyssen Wen
Date         : 2022-12-23 17:42:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 23:44:56
Description  : file content
FilePath     : /SVTAS/svtas/utils/cam/builder.py
'''
from . import target_utils as custom_model_targets
from ..package_utils import is_pytorch_grad_cam_available

if is_pytorch_grad_cam_available():
    from pytorch_grad_cam.utils import model_targets
else:
    raise ImportError()

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