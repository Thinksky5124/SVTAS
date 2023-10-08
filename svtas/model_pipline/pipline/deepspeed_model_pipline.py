'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:27:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 21:10:18
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/deepspeed_model_pipline.py
'''
import torch
from .torch_model_pipline import TorchModelPipline
from svtas.utils import AbstractBuildFactory
import deepspeed

@AbstractBuildFactory.register('model_pipline')
class DeepspeedModelPipline(TorchModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 device=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None,
                 ds_config={}) -> None:
        super().__init__(model, post_processing, device, criterion, optimizer,
                         lr_scheduler, pretrained)
        self.ds_config = ds_config
        self.ds_config['local_rank'] = self.local_rank
        deepspeed.init_distributed()
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(args = ds_config,
                             model = self.model,
                             optimizer = self.optimizer,
                             model_parameters = self.model.parameters(),
                             lr_scheduler = self.lr_scheduler)
