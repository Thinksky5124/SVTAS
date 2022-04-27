'''
Author: Thyssen Wen
Date: 2022-04-27 15:30:52
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 15:30:52
Description: build tools
FilePath: /ETESVS/utils/build.py
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