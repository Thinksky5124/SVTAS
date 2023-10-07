'''
Author: Thyssen Wen
Date: 2022-04-27 15:30:52
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 15:30:52
Description: build tools
FilePath: /ETESVS/utils/build.py
'''
import abc
from threading import RLock
from typing import List, AnyStr, Dict
# Refence:https://github.com/Thinksky5124/PaddleVideo/blob/develop/paddlevideo/utils/registry.py


class Registry(object):
    """
    The registry that provides name -> object mapping, to support third-party users' custom modules.
    To register an object:
    .. code-block:: python
        BACKBONES = Registry('backbone')
        @AbstractBuildFactory.register('model')
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
    
    @property
    def name(self):
        return self._name

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
        return obj

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

class BaseBuildFactory(metaclass=abc.ABCMeta):
    def __init__(self, registry_table = None):
        self.registry_table = registry_table
    
    @staticmethod
    def get_registry_table(object_type_name):
        for registry_table_name, registry_table in AbstractBuildFactory.REGISTRY_MAP.items():
            if object_type_name in registry_table:
                return registry_table
        raise KeyError("No object named '{}' found in all registry!".format(object_type_name))
    
    @abc.abstractmethod
    def create(self, *args, **kwargs):
        raise NotImplementedError("You must implement create function!")

class FromArgsBuildFactory(BaseBuildFactory):
    def build_from_args(self, object_type_name, *args, **kwargs):
        """Build a module from args.
        Args:
            registry (XXX): The registry to search the type from.
            key (str): the key.
        Returns:
            obj: The constructed object.
        """

        if self.registry_table is None:
            self.registry_table = self.get_registry_table(object_type_name)

        obj_cls = self.registry_table.get(object_type_name)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                    object_type_name, self.registry_table.name))

        return obj_cls(*args, **kwargs)
    
    def create(self, object_type_name, *args, **kwargs):
        return self.build_from_args(object_type_name, *args, **kwargs)
    
class FromConfigBuildFactory(BaseBuildFactory):
    def build_from_cfg(self, key='name', cfg = None, *args, **kwargs):
        """Build a module from config dict.
        Args:
            cfg (dict): Config dict. It should at least contain the key.
            registry (XXX): The registry to search the type from.
            key (str): the key.
        Returns:
            obj: The constructed object.
        """
        if cfg is None:
            return None
        assert isinstance(cfg, dict) and key in cfg, "Not specify class name"

        cfg_copy = cfg.copy()
        obj_type = cfg_copy.pop(key)

        if self.registry_table is None:
            self.registry_table = self.get_registry_table(obj_type)

        obj_cls = self.registry_table.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                    obj_type, self.registry_table.name))

        return obj_cls(*args, **cfg_copy)
    
    def create(self, cfg, *args, key='name', **kwargs):
        return self.build_from_cfg(key, cfg, *args, **kwargs)

class FromConfigWithSBPBuildFactory(FromConfigBuildFactory):
    def build_from_cfg(self, key='name', cfg = None, *args, **kwargs):
        """Build a module from config dict.
        Args:
            cfg (dict): Config dict. It should at least contain the key.
            registry (XXX): The registry to search the type from.
            key (str): the key.
        Returns:
            obj: The constructed object.
        """
        if cfg is None:
            return None
        assert isinstance(cfg, dict) and key in cfg

        cfg_copy = cfg.copy()
        obj_type = cfg_copy.pop(key)

        if self.registry_table is None:
            self.registry_table = self.get_registry_table(obj_type)
            
        obj_cls = self.registry_table.get(obj_type)
        if obj_cls is None:
            raise KeyError('{} is not in the {} registry'.format(
                    obj_type, self.registry_table.name))
        
        from .sbp import StochasticBackPropagation
        sbp_kwargs=dict()
        for key in StochasticBackPropagation.SBP_ARGUMENTS:
            if key in cfg_copy.keys():
                sbp_kwargs[key] = cfg_copy.pop(key)
                
        if 'sbp_build' in sbp_kwargs.keys() and sbp_kwargs['sbp_build']:
            sbp = StochasticBackPropagation(**sbp_kwargs)
            return sbp.register_module_from_instance(obj_cls, cfg_copy)

        return obj_cls(*args, **cfg_copy)
    
class AbstractBuildFactory(metaclass=abc.ABCMeta):
    BUILD_METHOD_MAP = dict(
        from_config = FromConfigBuildFactory,
        from_config_with_sbp = FromConfigWithSBPBuildFactory,
        from_args = FromArgsBuildFactory
    )
    REGISTRY_MAP: Dict = dict()
    single_lock = RLock()
    
    def __init__(self):
        pass

    # singleton design
    def __new__(cls, registory_name = None, build_method = "from_config", *args, **kwargs):
        if registory_name is None:
            with AbstractBuildFactory.single_lock:
                if not hasattr(AbstractBuildFactory, "_instance"):
                    AbstractBuildFactory._instance = object.__new__(cls)

            return AbstractBuildFactory._instance
        else:
            assert build_method in AbstractBuildFactory.BUILD_METHOD_MAP.keys(), f"Unsupport build_method: {build_method}!"
            assert registory_name in AbstractBuildFactory.REGISTRY_MAP.keys(), f"Unsupport registory_name: {registory_name}!"
            if registory_name is None:
                return AbstractBuildFactory.BUILD_METHOD_MAP[build_method]()
            return AbstractBuildFactory.BUILD_METHOD_MAP[build_method](AbstractBuildFactory.REGISTRY_MAP[registory_name])
    
    @staticmethod
    @property
    def registry_keys() -> List[AnyStr]:
        return list(AbstractBuildFactory.REGISTRY_MAP.keys())
    
    @staticmethod
    def get_registry_table(registry_name) -> List[AnyStr]:
        return list(AbstractBuildFactory.REGISTRY_MAP[registry_name].keys())
    
    @staticmethod
    def create_factory(registory_name = None, build_method = "from_config"):
        assert build_method in AbstractBuildFactory.BUILD_METHOD_MAP.keys(), f"Unsupport build_method: {build_method}!"
        assert registory_name in AbstractBuildFactory.REGISTRY_MAP.keys(), f"Unsupport registory_name: {registory_name}!"
        return AbstractBuildFactory.BUILD_METHOD_MAP[build_method](AbstractBuildFactory.REGISTRY_MAP[registory_name]) 

    @staticmethod
    def register(registory_name: str):
        """
        Register a class from `registory_name`, can be used as decorate
        .. code-block:: python
            @AbstractBuildFactory.register('backbone')
            class ResNet:
                pass
                
            build_factory_1 = AbstractBuildFactory('backbone')
            abstract_build_factory_2 = AbstractBuildFactory()
            build_factory_2 = abstract_build_factory_2.create_factory('backbone')
            a = build_factory_1.create(...)
            a = build_factory_2.create(...)
        """
        assert isinstance(registory_name, str), "registory_name must be a string!"
        if registory_name not in AbstractBuildFactory.REGISTRY_MAP.keys():
            AbstractBuildFactory.REGISTRY_MAP[registory_name] = Registry(registory_name)
        
        Registry_Class: Registry = AbstractBuildFactory.REGISTRY_MAP[registory_name]
        def actually_register(obj=None, name=None):
            return Registry_Class.register(obj, name)
        
        return actually_register
