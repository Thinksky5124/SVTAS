'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:14:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:50:34
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/base_pipline.py
'''
import abc
from typing import Any, Dict
from svtas.utils import AbstractBuildFactory
from ..wrapper import BaseModel
from svtas.optimizer import BaseLRScheduler, TorchOptimizer

class BaseModelPipline(metaclass=abc.ABCMeta):
    model: BaseModel
    criterion: BaseModel
    optimizer: TorchOptimizer
    lr_scheduler: BaseLRScheduler
    post_processing: None

    def __init__(self,
                 model,
                 post_processing,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None) -> None:
        def set_property(name: str, value: Dict, build_name: str = None):
            if isinstance(value, dict):
                setattr(self, name, AbstractBuildFactory.create_factory(build_name).create(value))
            else:
                setattr(self, name, value)

        self.model = AbstractBuildFactory.create_factory('architecture').create(model, key='architecture')
        self.criterion = AbstractBuildFactory.create_factory('loss').create(criterion)
        self.post_processing = AbstractBuildFactory.create_factory('post_processing').create(post_processing)

        # construct optimizer
        if isinstance(optimizer, dict):
            optimizer['model'] = self.model
            self.optimizer = AbstractBuildFactory.create_factory('optimizer').create(optimizer)
        else:
            self.optimizer = optimizer

        # construct lr_scheduler
        if isinstance(lr_scheduler, dict):
            lr_scheduler['optimizer'] = self.optimizer
            self.lr_scheduler = AbstractBuildFactory.create_factory('lr_scheduler').create(lr_scheduler)
        else:
            self.lr_scheduler = lr_scheduler

    @abc.abstractmethod
    def pre_froward(self, data_dict):
        return data_dict
    
    @abc.abstractmethod
    def forward(self, data_dict):
        raise NotImplementedError("You must implement forward function!")
    
    @abc.abstractmethod
    def after_forward(self, mode, end_info_dict):
        return end_info_dict
    
    @abc.abstractmethod
    def caculate_loss(self, loss_dict) -> dict:
        raise NotImplementedError("You must implement caculate_loss function!")
    
    @abc.abstractmethod
    def pre_backward(self, loss_dict):
        return loss_dict
    
    @abc.abstractmethod
    def backward(self, loss_dict):
        raise NotImplementedError("You must implement backward function!")
    
    @abc.abstractmethod
    def after_backward(self, loss_dict):
        return loss_dict
    
    @abc.abstractmethod
    def update_model_param(self):
        raise NotImplementedError("You must implement update_model_param function!")
    
    @abc.abstractmethod
    def init_model_param(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def resert_model_pipline(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def end_model_pipline(self):
        pass
    
    @abc.abstractmethod
    def save(self) -> Dict:
        """
        Return model param dict readly to save
        """
        raise NotImplementedError("You must implement save_model function!")

    @abc.abstractmethod
    def load(self, param_dict: Dict) -> None:
        raise NotImplementedError("You must implement load_model function!")

    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("You must implement __call__ function!")
