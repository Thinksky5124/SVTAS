'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:37:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 20:59:22
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/base_iter_method.py
'''
import abc
import time
from typing import Any, Dict, List
from svtas.utils.logger import BaseLogger, BaseRecord
from svtas.model_pipline import BaseModelPipline
from svtas.loader.dataloader import BaseDataloader
from svtas.metric import BaseMetric

class BaseIterMethod(metaclass=abc.ABCMeta):
    dataloader: BaseDataloader
    model_pipline: BaseModelPipline
    logger_dict: Dict[str, BaseLogger]
    record: BaseRecord
    metric: Dict[str, BaseMetric]

    def __init__(self,
                 batch_size: int,
                 mode: str,
                 criterion_metric_name: str,
                 save_interval: int = 10,
                 test_interval: int = -1) -> None:
        self.pass_check: bool = False
        self._mode = mode
        self.test_interval = test_interval
        self.save_interval = save_interval
        self.batch_size = batch_size
        self.hook_dict: Dict[str, List] = dict()
        self.criterion_metric_name = criterion_metric_name
        self.memory_score = 0.0

    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, val):
        assert val in ['train', 'test', 'validation', 'infer', 'profile', 'visulaize'], f"Unsupport mode val: {val}!"
        self._mode = val

    def init_iter_method(self,
                  logger_dict: Dict[str, BaseLogger],
                  record: BaseRecord,
                  metric: Dict[str, BaseMetric],
                  model_pipline: BaseModelPipline):
        self.logger_dict = logger_dict
        self.record = record
        self.metric = metric
        self.model_pipline = model_pipline
    
    def register_hook(self, key, func):
        """
        You should not modify value in hook function!
        """
        assert key in ['epoch_pre', 'iter_pre', 'iter_end', 'epoch_end', 'every_iter_end', 'every_batch_end']
        if key not in self.hook_dict:
            self.hook_dict[key] = [func]
        else:
            self.hook_dict[key].append(func)
    
    def exec_hook(self, key, *args, **kwargs):
        if key in self.hook_dict:
            for func in self.hook_dict[key]:
                func(*args, **kwargs)
    
    def register_test_hook(self, func):
        """
        You should not modify value in hook function!

        hook function:
        ```python
        def hook(best_score: float) -> float:
            ...
            test_engine.run()
            return test_engine.iter_method.best_score
        ```
        """
        self.hook_dict['test_hook'] = func

    def test_hook(self, best_score) -> float:
        return self.hook_dict['test_hook'](best_score)

    def set_dataloader(self, dataloader: BaseDataloader):
        self.dataloader = dataloader
    
    def run_one_forward(self, data_dict):
        outputs, loss_dict = self.model_pipline(data_dict)
        return outputs, loss_dict

    def run_check(self) -> bool:
        """
        This function excute before run api to check all component for running readly
        """
        assert hasattr(self, "model_pipline"), "Unpass `run_check`, please excute api `init_iter_method` before run or prepare other initialize!"
        return True

    @abc.abstractmethod
    def init_run(self):
        self.best_score = 0.0

    @abc.abstractmethod
    def run(self) -> float:
        pass
    
    @abc.abstractmethod
    def end_run(self):
        pass
    
    @abc.abstractmethod
    def end(self):
        pass

    @abc.abstractmethod
    def save(self) -> Dict:
        save_dict = dict()
        return save_dict
    
    @abc.abstractmethod
    def load(self, load_dict: Dict) -> None:
        pass