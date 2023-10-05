'''
Author       : Thyssen Wen
Date         : 2023-10-05 19:08:55
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 21:02:21
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/optim/base_optim.py
'''
import abc
from svtas.utils.logger import get_logger
from torch.optim import Optimizer

def log_params(filter_params, name):
    if len(filter_params.keys()) > 0:
        keys_str = ','.join(filter_params.keys())
        logger = get_logger()
        logger.info(name + ': ' + keys_str)
        logger.info('='*50)

def filter_normal_optim_params(params, no_main, need_log=False):
    filter_params = {}
    for n, p in params:
        if p.requires_grad:
            if not any(nd in n for nd in no_main):
                filter_params[n] = p
    if need_log:
        log_params(filter_params=filter_params, name="normal_optim_params")
    return list(filter_params.values())

def filter_no_decay_optim_params(params, finetuning_key, no_decay_key, need_log=False):
    filter_params = {}
    for n, p in params:
        if p.requires_grad:
            if not any(nd in n for nd in finetuning_key) and any(nd in n for nd in no_decay_key):
                filter_params[n] = p
    if need_log:
        log_params(filter_params=filter_params, name="no_decay_optim_params")
    return list(filter_params.values())

def filter_no_decay_finetuning_optim_params(params, finetuning_key, no_decay_key, need_log=False):
    filter_params = {}
    for n, p in params:
        if p.requires_grad:
            if any(nd in n for nd in finetuning_key) and any(nd in n for nd in no_decay_key):
                filter_params[n] = p
    if need_log:
        log_params(filter_params=filter_params, name="no_decay_finetuning_optim_params")
    return list(filter_params.values())

def filter_finetuning_optim_params(params, finetuning_key, no_decay_key, need_log=False):
    filter_params = {}
    for n, p in params:
        if p.requires_grad:
            if any(nd in n for nd in finetuning_key) and not any(nd in n for nd in no_decay_key):
                filter_params[n] = p
    if need_log:
        log_params(filter_params=filter_params, name="finetuning_optim_params")
    return list(filter_params.values())

class BaseOptimizer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def step(self, closure):
        pass
    
    @abc.abstractmethod
    def get_optim_policies(self, model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay):
        pass

class TorchOptimizer(Optimizer, BaseOptimizer):
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