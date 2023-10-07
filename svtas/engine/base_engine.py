'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:28:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 20:45:33
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

        for name, cfg in metric.items():
            if isinstance(cfg, dict):
                self.metric[name] = AbstractBuildFactory.create_factory('metric').create(cfg)
            else:
                self.metric = metric

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
        assert val in ['train', 'test', 'validation', 'infer', 'profile', 'visulaize', 'extract']
        # set running mode
        self._running_mode = val
        self.iter_method.mode = self.running_mode
        if self.running_mode == 'train':
            self.model_pipline.train()
        else:
            self.model_pipline.eval()
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

@AbstractBuildFactory.register('engine')
class BaseImplementEngine(BaseEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 running_mode = 'train') -> None:
        super().__init__(model_name,
                         model_pipline,
                         logger_dict,
                         record,
                         metric,
                         iter_method,
                         checkpointor,
                         running_mode)

    def init_engine(self, dataloader: BaseDataloader = None):
        if dataloader is not None:
            self.set_dataloader(dataloader=dataloader)
        self.iter_method.init_iter_method(logger_dict=self.logger_dict,
                                          record=self.record,
                                          metric=self.metric,
                                          model_pipline=self.model_pipline)
        self.model_pipline.to(device=self.model_pipline.device)
        self.record.init_record()
        # set running mode
        self.iter_method.mode = self.running_mode
        if self.running_mode == 'train':
            self.model_pipline.train()
        else:
            self.model_pipline.eval()

    def set_dataloader(self, dataloader: BaseDataloader):
        self.iter_method.set_dataloader(dataloader=dataloader)
    
    def resume_impl(self, load_dict: Dict):
        self.model_pipline.load(load_dict['model_pipline'])
        self.iter_method.load(load_dict['iter_method'])
        self.record.load(load_dict['record'])

    def resume(self, path: str = None):
        if self.checkpointor.load_flag and path is None:
            load_dict = self.checkpointor.load()
            for key, logger in self.logger_dict.items():
                logger.info(f"resume engine from checkpoint file: {self.checkpointor.load_path}")
        elif path is not None:
            load_dict = self.checkpointor.load(path)
            for key, logger in self.logger_dict.items():
                logger.info(f"resume engine from checkpoint file: {path}")
        else:
            raise FileNotFoundError("You must specify a valid path!")
        self.resume_impl(load_dict)

    def save(self, save_dict: Dict = {}, path: str = None, file_name: str = None):
        save_dict['model_pipline'] = self.model_pipline.save()
        save_dict['iter_method'] = self.iter_method.save()
        save_dict['record'] = self.record.save()
        if self.checkpointor.save_flag and path is None:
            self.checkpointor.save(save_dict = save_dict, file_name = file_name)
        elif path is not None:
            self.checkpointor.save(save_dict = save_dict, path = path, file_name = file_name)
        else:
            raise FileNotFoundError("You must specify a valid path!")

    def run(self):
        for epoch in self.iter_method.run():
            if self.running_mode in ['train']:
                self.save(file_name = self.model_name + f"_epoch_{epoch + 1:05d}")
            elif self.running_mode in ['validation']:
                self.save(file_name = self.model_name + "_best")
        
    def shutdown(self):
        self.model_pipline.end_model_pipline()