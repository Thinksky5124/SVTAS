'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:14:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 16:18:56
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/base_pipline.py
'''
import os
import abc
from typing import Any, Dict, List
from svtas.utils import AbstractBuildFactory
from ..wrapper import BaseModel
from svtas.optimizer import BaseLRScheduler, TorchOptimizer

class BaseModelPipline(metaclass=abc.ABCMeta):
    model: BaseModel
    criterion: BaseModel
    optimizer: TorchOptimizer
    lr_scheduler: BaseLRScheduler
    post_processing: None
    local_rank: int
    world_size: int

    def __init__(self,
                 model,
                 post_processing,
                 device,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None) -> None:
        self.model = AbstractBuildFactory.create_factory('model').create(model)
        self.criterion = AbstractBuildFactory.create_factory('loss').create(criterion)
        self.post_processing = AbstractBuildFactory.create_factory('post_processing').create(post_processing)
        self._device = device
        self.pretrained = pretrained

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
        
        if self.lr_scheduler is not None and self.optimizer is not None and self.criterion is not None:
            self.train()
        else:
            self.eval()
        
        if self.pretrained is not None:
            self.load_from_ckpt_file()
        
        # prepare for distribution train
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])

    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, val: bool):
        self._training = val

    def train(self):
        self._training = True
        self.model.training = True

    def eval(self):
        self.model.eval()
        self._training = False
    
    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = device
        self.model.to(device)
        if self.criterion is not None:
            self.criterion.to(device)

    @abc.abstractmethod
    def forward(self, data_dict):
        raise NotImplementedError("You must implement forward function!")
    
    @abc.abstractmethod
    def caculate_loss(self, loss_dict) -> Dict:
        raise NotImplementedError("You must implement caculate_loss function!")
    
    @abc.abstractmethod
    def init_post_processing(self, input_data: Dict) -> None:
        pass

    @abc.abstractmethod
    def update_post_processing(self, model_outputs: Dict, input_data: Dict) -> None:
        pass

    @abc.abstractmethod
    def output_post_processing(self, model_outputs: Dict = None, input_data: Dict = None) -> List:
        pass

    @abc.abstractmethod
    def backward(self, loss_dict):
        raise NotImplementedError("You must implement backward function!")
    
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
    def update_optim_policy(self):
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
    
    def load_from_ckpt_file_ckeck(self, ckpt_path: str = None):
        if ckpt_path is None:
            if self.pretrained is None:
                raise FileExistsError("Not ckpt file exits!")
            else:
                ckpt_path = self.pretrained
        return ckpt_path
    
    @abc.abstractmethod
    def load_from_ckpt_file(self, ckpt_path: str = None):
        ckpt_path = self.load_from_ckpt_file_ckeck(ckpt_path)

    @abc.abstractmethod
    def train_run(self, data_dict) -> Dict:
        pass
    
    @abc.abstractmethod
    def test_run(self, data_dict) -> Dict:
        pass

    def __call__(self, data_dict) -> Any:
        if self.training:
            return self.train_run(data_dict=data_dict)
        else:
            return self.test_run(data_dict=data_dict)

class FakeModelPipline(BaseModelPipline):
    def __init__(self, post_processing) -> None:
        super().__init__(None, post_processing, None)
    
    def forward(self, data_dict):
        pass
    
    def caculate_loss(self, loss_dict) -> Dict:
        pass
    
    def init_post_processing(self, input_data: Dict) -> None:
        pass

    def update_post_processing(self, model_outputs: Dict, input_data: Dict) -> None:
        pass

    def output_post_processing(self, model_outputs: Dict = None, input_data: Dict = None) -> List:
        pass

    def backward(self, loss_dict):
        pass
    
    def update_model_param(self):
        pass
    
    def init_model_param(self, *args, **kwargs):
        pass

    def resert_model_pipline(self, *args, **kwargs):
        pass

    def update_optim_policy(self):
        pass

    def end_model_pipline(self):
        pass
    
    def save(self) -> Dict:
        """
        Return model param dict readly to save
        """
        pass

    def load(self, param_dict: Dict) -> None:
        pass

    def train_run(self, data_dict) -> Dict:
        pass
    
    def test_run(self, data_dict) -> Dict:
        pass
    
    def __call__(self, data_dict) -> Any:
        return data_dict