'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:28:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-28 20:11:46
Description  : file content
FilePath     : /SVTAS/svtas/engine/base_engine.py
'''
import abc
from svtas.model_pipline import BaseModelPipline
from svtas.utils.logger import BaseLogger
from .iter_method import BaseIterMethod
from .checkpoint import BaseCheckpointor
from svtas.utils.logger import BaseRecord
from svtas.utils import AbstractBuildFactory
from typing import Dict
    
class BaseEngine(metaclass=abc.ABCMeta):
    model_pipline: BaseModelPipline
    logger: BaseLogger
    iter_method: BaseIterMethod
    checkpointor: BaseCheckpointor
    record: BaseRecord
    
    def __init__(self,
                 model_pipline: Dict,
                 logger: Dict,
                 record: Dict,
                 iter_method: Dict,
                 checkpointor: Dict) -> None:
        self.model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(model_pipline)
        self.logger = AbstractBuildFactory.create_factory('logger').create(logger)
        self.iter_method = AbstractBuildFactory.create_factory('engine_component').create(iter_method)
        self.checkpointor = AbstractBuildFactory.create_factory('engine_component').create(checkpointor)
        self.record = AbstractBuildFactory.create_factory('record').create(record)

    @abc.abstractmethod
    def init_engine(self):
        pass

    @abc.abstractmethod
    def resume(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def save(self, path: str = None):
        pass

    @abc.abstractmethod
    def run():
        raise NotImplementedError("You must implement run function!")
    
    @abc.abstractmethod
    def shutdown():
        raise NotImplementedError("You must implement shutdown function!")