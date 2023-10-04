'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:28:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-04 16:23:58
Description  : file content
FilePath     : /SVTAS/svtas/engine/base_engine.py
'''
import abc
from svtas.model_pipline import BaseModelPipline
from svtas.utils.logger import BaseLogger
from .iter_method import BaseIterMethod
from .checkpoint import BaseCheckpointor
from svtas.utils.logger import BaseRecord
from svtas.loader.dataloader import BaseDataloader
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
    def init_engine(self, dataloader: BaseDataloader):
        self.set_dataloader(dataloader=dataloader)
        self.iter_method.init_iter_method(logger=self.logger,
                                          record=self.record,
                                          model_pipline=self.model_pipline)

    @abc.abstractmethod
    def set_dataloader(self, dataloader: BaseDataloader):
        self.iter_method.set_dataloader(dataloader=dataloader)

    @abc.abstractmethod
    def resume(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def save(self, path: str = None):
        pass

    @abc.abstractmethod
    def run(self):
        self.iter_method.run()
    
    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError("You must implement shutdown function!")