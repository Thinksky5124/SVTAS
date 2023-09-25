'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:14:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-22 15:31:25
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/base_pipline.py
'''
import abc

class BaseModelPipline(metaclass=abc.ABCMeta):
    def __init__(self,
                 model,
                 criterion=None,
                 optimizer=None,) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer

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
    def before_backward(self, loss_dict):
        return loss_dict
    
    @abc.abstractmethod
    def backward(self, loss_dict):
        raise NotImplementedError("You must implement backward function!")
    
    @abc.abstractmethod
    def after_backward(self, loss_dict):
        return loss_dict
    
    @abc.abstractmethod
    def update_model(self):
        raise NotImplementedError("You must implement update_model function!")
    
    @abc.abstractmethod
    def init_model_param(self, *args, **kwargs):
        pass
    
    @abc.abstractmethod
    def save_model(self, path):
        pass
