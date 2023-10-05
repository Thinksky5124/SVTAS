'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:37:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 21:18:49
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
    logger: BaseLogger
    record: BaseRecord
    metric: BaseMetric

    def __init__(self, *args, **kwargs) -> None:
        self.pass_check: bool = False

    def init_iter_method(self,
                  logger: BaseLogger,
                  record: BaseRecord,
                  metric: BaseMetric,
                  model_pipline: BaseModelPipline):
        self.logger = logger
        self.record = record
        self.metric = metric
        self.model_pipline = model_pipline
        self.hook_dict: Dict[str, List] = dict()
    
    def register_hook(self, key, func):
        """
        You should not modify value in hook function!
        """
        if key not in self.hook_dict:
            self.hook_dict[key] = [func]
        else:
            self.hook_dict[key].append(func)
    
    def exec_hook(self, key, *args, **kwargs):
        for func in self.hook_dict[key]:
            func(*args, **kwargs)

    def set_dataloader(self, dataloader: BaseDataloader):
        self.dataloader = dataloader
    
    def run_one_forward(self, data_dict):
        score, loss_dict = self._run_model_pipline(data_dict)
        data_dict['score'] = score
        self.record.update_loss_dict(loss_dict)
        return loss_dict, data_dict

    def run_check(self) -> bool:
        """
        This function excute before run api to check all component for running readly
        """
        assert hasattr(self, "model_pipline"), "You must excute api `init_iter_method` before run!"
        return True

    @abc.abstractmethod
    def run(self, *args: Any, **kwds: Any) -> Any:
        assert self.run_check(), "Unpass `run_check`, please excute api `init_iter_method` before run or prepare other initialize!"
    
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