'''
Author       : Thyssen Wen
Date         : 2023-10-19 20:28:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-23 23:38:22
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch/base.py
'''
import abc
from typing import Any, Dict, List
from ...wrapper import BaseModel
from svtas.optimizer import BaseLRScheduler, TorchOptimizer
from ..base_pipline import BaseModelPipline
from svtas.utils import AbstractBuildFactory

class BaseTorchModelPipline(BaseModelPipline):
    criterion: BaseModel
    optimizer: TorchOptimizer
    lr_scheduler: BaseLRScheduler
    post_processing: None

    def __init__(self,
                 model,
                 post_processing,
                 device,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None) -> None:
        super().__init__(model, device)
        self.pretrained = pretrained
        self.model = AbstractBuildFactory.create_factory('model').create(self.model)
        self.criterion = AbstractBuildFactory.create_factory('loss').create(criterion)
        self.post_processing = AbstractBuildFactory.create_factory('post_processing').create(post_processing)
        # init model
        self.init_model_weight()

        if self.pretrained is not None:
            self.load_from_ckpt_file()
        
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

    def train(self):
        self._training = True
        self.model.train()

    def eval(self):
        self.model.eval()
        self._training = False
    
    def to(self, device):
        super().to(device=device)
        self.model.to(device)
        if self.criterion is not None:
            self.criterion.to(device)

    def init_model_weight(self, init_cfg: Dict = None) -> None:
        if init_cfg is None:
            self.model.init_weights()
        else:
            self.model.init_weights(init_cfg=init_cfg)
    
    def post_processing_is_init(self):
        if self.post_processing is not None:
            return self.post_processing.init_flag
        else:
            return False
    
    def set_post_processing_init_flag(self, val: bool):
        if self.post_processing is not None:
            self.post_processing.init_flag = val
    
    def resert_model_pipline(self, *args, **kwargs):
        self.model.reset_state()
        return super().resert_model_pipline(*args, **kwargs)
    
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
    def update_optim_policy(self):
        pass

class FakeTorchModelPipline(BaseTorchModelPipline):
    def __init__(self,
                 post_processing) -> None:
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