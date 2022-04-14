'''
Author: Thyssen Wen
Date: 2022-04-14 16:16:56
LastEditors: Thyssen Wen
LastEditTime: 2022-04-14 17:16:12
Description: registry and builder model
FilePath: /ETESVS/model/builder.py
'''
# Refence:https://github.com/open-mmlab/mmaction2/blob/f3d4817d781b45fa02447a2181db5c87eccc3335/mmaction/models/builder.py
# Refence:https://github.com/Thinksky5124/PaddleVideo/blob/develop/paddlevideo/utils/registry.py

class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.
    To register an object:
    .. code-block:: python
        BACKBONES = Registry('backbone')
        @BACKBONES.register()
        class ResNet:
            pass
    Or:
    .. code-block:: python
        BACKBONES = Registry('backbone')
        class ResNet:
            pass
        BACKBONES.register(ResNet)
    Usage: To build a module.
    .. code-block:: python
        backbone_name = "ResNet"
        b = BACKBONES.get(backbone_name)()
    """
    def __init__(self, name):
        """
        Args:
            name (str): the name of this registry
        """
        self._name = name
        self._obj_map = {}

    def __contains__(self, key):
        return self._obj_map.get(key) is not None

    def _do_register(self, name, obj):
        assert (
            name not in self._obj_map
        ), "An object named '{}' was already registered in '{}' registry!".format(
            name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None, name=None):
        """
        Register the given object under the the name `obj.__name__`.
        Can be used as either a decorator or not. See docstring of this class for usage.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class, name=name):
                if name is None:
                    name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class

            return deco

        # used as a function call
        if name is None:
            name = obj.__name__
        self._do_register(name, obj)

    def get(self, name):
        """Get the registry record.
        Args:
            name (str): The class name.
        Returns:
            ret: The class.
        """
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError(
                "No object named '{}' found in '{}' registry!".format(
                    name, self._name))

        return ret

BACKBONES = Registry('backbone')
NECKS = Registry('neck')
HEADS = Registry('head')
ARCHITECTURE = Registry('architecture')
LOSSES = Registry('loss')
                       
def build(cfg, registry, key='name'):
    """Build a module from config dict.
    Args:
        cfg (dict): Config dict. It should at least contain the key.
        registry (XXX): The registry to search the type from.
        key (str): the key.
    Returns:
        obj: The constructed object.
    """

    assert isinstance(cfg, dict) and key in cfg

    cfg_copy = cfg.copy()
    obj_type = cfg_copy.pop(key)

    obj_cls = registry.get(obj_type)
    if obj_cls is None:
        raise KeyError('{} is not in the {} registry'.format(
                obj_type, registry.name))
    return obj_cls(**cfg_copy)
    

def build_backbone(cfg):
    """Build backbone."""
    return build(cfg, BACKBONES)

def build_head(cfg):
    """Build head."""
    return build(cfg, HEADS)

def build_neck(cfg):
    """Build neck."""
    return build(cfg, NECKS)

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
