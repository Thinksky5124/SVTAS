'''
Author       : Thyssen Wen
Date         : 2023-10-11 19:17:10
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 20:30:38
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/torch_utils/layer/build_utils.py
'''
import inspect
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union, Tuple, Optional

from svtas.utils import AbstractBuildFactory

import inspect
from typing import Dict, Tuple, Union

import torch.nn as nn
from svtas.utils import AbstractBuildFactory

AbstractBuildFactory.register_obj(registory_name='model', obj_name='BN', obj=nn.BatchNorm2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='BN1d', obj=nn.BatchNorm1d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='BN2d', obj=nn.BatchNorm2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='BN3d', obj=nn.BatchNorm3d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='GN', obj=nn.GroupNorm)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='LN', obj=nn.LayerNorm)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='IN', obj=nn.InstanceNorm2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='IN1d', obj=nn.InstanceNorm1d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='IN2d', obj=nn.InstanceNorm2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='IN3d', obj=nn.InstanceNorm3d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='Conv1d', obj=nn.Conv1d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='Conv2d', obj=nn.Conv2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='Conv3d', obj=nn.Conv3d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='Conv', obj=nn.Conv2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='zero', obj=nn.ZeroPad2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='reflect', obj=nn.ReflectionPad2d)
AbstractBuildFactory.register_obj(registory_name='model', obj_name='replicate', obj=nn.ReplicationPad2d)

def infer_abbr(class_type):
    """Infer abbreviation from the class name.

    When we build a norm layer with `build_norm_layer()`, we want to preserve
    the norm type in variable names, e.g, self.bn1, self.gn. This method will
    infer the abbreviation to map class types to abbreviations.

    Rule 1: If the class has the property "_abbr_", return the property.
    Rule 2: If the parent class is _BatchNorm, GroupNorm, LayerNorm or
    InstanceNorm, the abbreviation of this layer will be "bn", "gn", "ln" and
    "in" respectively.
    Rule 3: If the class name contains "batch", "group", "layer" or "instance",
    the abbreviation of this layer will be "bn", "gn", "ln" and "in"
    respectively.
    Rule 4: Otherwise, the abbreviation falls back to "norm".

    Args:
        class_type (type): The norm layer type.

    Returns:
        str: The inferred abbreviation.
    """
    if not inspect.isclass(class_type):
        raise TypeError(
            f'class_type must be a type, but got {type(class_type)}')
    if hasattr(class_type, '_abbr_'):
        return class_type._abbr_
    elif issubclass(class_type, nn.GroupNorm):
        return 'gn'
    elif issubclass(class_type, nn.LayerNorm):
        return 'ln'
    else:
        class_name = class_type.__name__.lower()
        if 'batch' in class_name:
            return 'bn'
        elif 'group' in class_name:
            return 'gn'
        elif 'layer' in class_name:
            return 'ln'
        elif 'instance' in class_name:
            return 'in'
        else:
            return 'norm_layer'
        
def build_activation_layer(cfg: Dict) -> nn.Module:
    """Build activation layer.

    Args:
        cfg (dict): The activation layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate an activation layer.

    Returns:
        nn.Module: Created activation layer.
    """
    return AbstractBuildFactory.create_factory('model').create(cfg)

def build_norm_layer(cfg: Dict,
                     num_features: int,
                     postfix: Union[int, str] = '') -> Tuple[str, nn.Module]:
    """Build normalization layer.

    Args:
        cfg (dict): The norm layer config, which should contain:

            - type (str): Layer type.
            - layer args: Args needed to instantiate a norm layer.
            - requires_grad (bool, optional): Whether stop gradient updates.
        num_features (int): Number of input channels.
        postfix (int | str): The postfix to be appended into norm abbreviation
            to create named layer.

    Returns:
        tuple[str, nn.Module]: The first element is the layer name consisting
        of abbreviation and postfix, e.g., bn1, gn. The second element is the
        created norm layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')

    if inspect.isclass(layer_type):
        norm_layer = layer_type
    else:
        if norm_layer is None:
            raise KeyError(f'Cannot find {norm_layer} in registry under '
                           f'scope name model')
    abbr = infer_abbr(norm_layer)

    assert isinstance(postfix, (int, str))
    name = abbr + str(postfix)

    requires_grad = cfg_.pop('requires_grad', True)
    cfg_.setdefault('eps', 1e-5)
    if norm_layer is not nn.GroupNorm:
        layer = norm_layer(num_features, **cfg_)
        if layer_type == 'SyncBN' and hasattr(layer, '_specify_ddp_gpu_num'):
            layer._specify_ddp_gpu_num(1)
    else:
        assert 'num_groups' in cfg_
        layer = norm_layer(num_channels=num_features, **cfg_)

    for param in layer.parameters():
        param.requires_grad = requires_grad

    return name, layer

def build_conv_layer(cfg: Optional[Dict], *args, **kwargs) -> nn.Module:
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if inspect.isclass(layer_type):
        return layer_type(*args, **kwargs, **cfg_)  # type: ignore
    # Switch registry to the target scope. If `conv_layer` cannot be found
    # in the registry, fallback to search `conv_layer` in the
    # mmengine.MODELS.
    conv_layer = AbstractBuildFactory.REGISTRY_MAP['model'].get(layer_type)
    if conv_layer is None:
        raise KeyError(f'Cannot find {conv_layer} in registry under scope '
                       f'name model')
    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


def build_padding_layer(cfg: Dict, *args, **kwargs) -> nn.Module:
    """Build padding layer.

    Args:
        cfg (dict): The padding layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate a padding layer.

    Returns:
        nn.Module: Created padding layer.
    """
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')

    cfg_ = cfg.copy()
    padding_type = cfg_.pop('type')
    if inspect.isclass(padding_type):
        return padding_type(*args, **kwargs, **cfg_)
    # Switch registry to the target scope. If `padding_layer` cannot be found
    # in the registry, fallback to search `padding_layer` in the
    # mmengine.MODELS.
    padding_layer = AbstractBuildFactory.REGISTRY_MAP['model'].get(padding_type)
    if padding_layer is None:
        raise KeyError(f'Cannot find {padding_layer} in registry under scope '
                       f'name model')
    layer = padding_layer(*args, **kwargs, **cfg_)

    return layer