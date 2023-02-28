'''
Author       : Thyssen Wen
Date         : 2023-02-27 21:05:03
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-28 10:36:55
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/optim/helper_function.py
'''
from ...utils.logger import get_logger

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