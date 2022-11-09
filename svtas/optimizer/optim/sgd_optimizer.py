'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 21:03:47
Description  : SGD optimizer
FilePath     : /SVTAS/svtas/optimizer/optim/sgd_optimizer.py
'''
from ..builder import OPTIMIZER
import torch

@OPTIMIZER.register()
class SGDOptimizer(torch.optim.SGD):
    def __init__(self,
                 model,
                 learning_rate=0.01,
                 momentum=0.9,
                 weight_decay=1e-4,
                 finetuning_scale_factor=0.1,
                 no_decay_key = [],
                 finetuning_key = [],
                 freeze_key = [],
                 **kwargs) -> None:
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super().__init__(params=params, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    
    def get_optim_policies(self, model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay):
        params = list(model.named_parameters())
        no_main = no_decay_key + finetuning_key

        for n, p in params:
            for nd in freeze_key:
                if nd in n:
                    p.requires_grad = False

        normal_optim_params = filter(lambda p : p.requires_grad, [p for n,p in params if not any(nd in n for nd in no_main)])
        no_decay_optim_params = filter(lambda p : p.requires_grad, [p for n,p in params if not any(nd in n for nd in finetuning_key) and any(nd in n for nd in no_decay_key)])
        no_decay_finetuning_optim_params = filter(lambda p : p.requires_grad, [p for n,p in params if any(nd in n for nd in finetuning_key) and any(nd in n for nd in no_decay_key) ])
        finetuning_optim_params = filter(lambda p : p.requires_grad, [p for n,p in params if any(nd in n for nd in finetuning_key) and not any(nd in n for nd in no_decay_key)])

        param_group = [
            {'params':normal_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate},
            {'params':no_decay_optim_params, 'weight_decay':0, 'lr':learning_rate},
            {'params':no_decay_finetuning_optim_params, 'weight_decay':0, 'lr':learning_rate * finetuning_scale_factor},
            {'params':finetuning_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate * finetuning_scale_factor}
        ]
        return param_group