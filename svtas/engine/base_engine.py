'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:28:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 18:28:16
Description  : file content
FilePath     : /SVTAS/svtas/engine/base_engine.py
'''
import abc
from svtas.model_pipline import BaseModelPipline
from svtas.utils.logger import BaseLogger, get_logger
from .iter_method import BaseIterMethod
from .checkpoint import BaseCheckpointor
from svtas.utils.logger import BaseRecord
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory
from svtas.metric import BaseMetric
from typing import Dict, List
    
class BaseEngine(metaclass=abc.ABCMeta):
    model_pipline: BaseModelPipline
    logger_dict: Dict[str, BaseLogger]
    iter_method: BaseIterMethod
    checkpointor: BaseCheckpointor
    record: BaseRecord
    metric: BaseMetric

    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 running_mode = 'train') -> None:
        def set_property(name: str, value: Dict, build_name: str = None):
            if isinstance(value, dict):
                setattr(self, name, AbstractBuildFactory.create_factory(build_name).create(value))
            else:
                setattr(self, name, value)

        set_property('model_pipline', model_pipline, build_name='model_pipline')
        set_property('iter_method', iter_method, build_name='engine_component')
        set_property('checkpointor', checkpointor, build_name='engine_component')
        set_property('record', record, build_name='record')

        if metric is not None and len(metric) > 0:
            for name, cfg in metric.items():
                if isinstance(cfg, dict):
                    self.metric[name] = AbstractBuildFactory.create_factory('metric').create(cfg)
                else:
                    self.metric = metric
        else:
            self.metric = None

        self.logger_dict = {}
        for class_name, cfg in logger_dict.items():
            if isinstance(cfg, dict):
                self.logger_dict[cfg['name']] = get_logger(cfg['name'])
            else:
                self.logger_dict = logger_dict
        self.model_name = model_name
        self._running_mode = running_mode

    @property
    def running_mode(self):
        return self._running_mode
    
    @running_mode.setter
    def running_mode(self, val: str):
        assert val in ['train', 'test', 'validation', 'infer', 'profile', 'visulaize', 'extract', 'export']
        # set running mode
        self._running_mode = val
        if self.iter_method is not None:
            self.iter_method.mode = self.running_mode
        if self.running_mode == 'train':
            self.model_pipline.train()
        else:
            self.model_pipline.eval()
            if 'lr' in self.record:
                self.record.remove_one_record('lr')
    
    @abc.abstractmethod
    def init_engine(self, dataloader: BaseDataloader = None):
        pass

    @abc.abstractmethod
    def set_dataloader(self, dataloader: BaseDataloader):
        pass
    
    @abc.abstractmethod
    def resume_impl(self, load_dict: Dict):
        pass

    @abc.abstractmethod
    def resume(self, path: str = None):
        pass

    @abc.abstractmethod
    def save(self, save_dict: Dict, path: str = None):
        pass

    @abc.abstractmethod
    def run(self):
        pass
    
    @abc.abstractmethod
    def shutdown(self):
        pass
