'''
Author       : Thyssen Wen
Date         : 2022-05-06 16:05:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-27 21:06:18
Description  : adam optimizer
FilePath     : /SVTAS/svtas/optimizer/optim/adam_optimizer.py
'''
from ..builder import OPTIMIZER
import torch
from .helper_function import (filter_normal_optim_params, filter_no_decay_optim_params,
                              filter_no_decay_finetuning_optim_params, filter_finetuning_optim_params)

@OPTIMIZER.register()
class AdamOptimizer(torch.optim.Adam):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 betas=(0.9, 0.999),
                 weight_decay=1e-4,
                 finetuning_scale_factor=0.1,
                 no_decay_key = [],
                 finetuning_key = [],
                 freeze_key = [],
                 **kwargs) -> None:
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super().__init__(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    
    def get_optim_policies(self, model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay):
        params = list(model.named_parameters())
        no_main = no_decay_key + finetuning_key

        for n, p in params:
            for nd in freeze_key:
                if nd in n:
                    p.requires_grad = False

        normal_optim_params = filter_normal_optim_params(params=params, no_main=no_main)
        no_decay_optim_params = filter_no_decay_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)
        no_decay_finetuning_optim_params = filter_no_decay_finetuning_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)
        finetuning_optim_params = filter_finetuning_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)

        param_group = [
            {'params':normal_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate},
            {'params':no_decay_optim_params, 'weight_decay':0, 'lr':learning_rate},
            {'params':no_decay_finetuning_optim_params, 'weight_decay':0, 'lr':learning_rate * finetuning_scale_factor},
            {'params':finetuning_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate * finetuning_scale_factor}
        ]
        return param_group