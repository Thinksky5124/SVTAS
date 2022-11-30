'''
Author       : Thyssen Wen
Date         : 2022-11-30 10:15:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-30 14:35:14
Description  : Hook Unit Test Class
FilePath     : /SVTAS/tests/common/hook.py
'''
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
from logging import getLogger
logger = getLogger('test')
from .check import check_tensor_close

class TorchHook(object):
    results_dict = dict()

    def __init__(self,
                 eps: float = 1e-6) -> None:
        self.eps = eps
    
    @staticmethod
    def backward_hook(Module: nn.Module, grad_in, grad_out):
        TorchHook.results_dict[Module._get_name() + '_grad_in'] = grad_in
        TorchHook.results_dict[Module._get_name() + '_grad_out'] = grad_out
        logger.info(f"Logging {Module._get_name()} backward grad!")
    
    @staticmethod
    def forward_hook(Module: nn.Module, fea_in, fea_out):
        TorchHook.results_dict[Module._get_name() + '_fea_in'] = fea_in
        TorchHook.results_dict[Module._get_name() + '_fea_out'] = fea_out
        logger.info(f"Logging {Module._get_name()} forward feature!")
    
    def register_backward_hook(self, Module: nn.Module):
        Module.register_backward_hook(TorchHook.backward_hook)
        logger.info(f"Register {Module._get_name()} backward hook!")
    
    def register_forward_hook(self, Module: nn.Module):
        Module.register_forward_hook(TorchHook.forward_hook)
        logger.info(f"Register {Module._get_name()}  forward hook!")
    
    def check_backward_w_torch_autograd(self, module: nn.Module, *test_input) -> bool:
        logger.info(f'Use automatic torch criterion check!')
        return torch.autograd.gradcheck(module, test_input, eps=self.eps)

    def check_w_criterion(self, test_criterion_dict: Dict[str, np.array]) -> Tuple[bool, float]:
        logger.info(f'Use manual criterion check!')
        all_close = False
        test_coverage = 0
        for k, v in test_criterion_dict.items():
            if k in TorchHook.results_dict.keys():
                i_tensors = TorchHook.results_dict[k]
                for i_tensor, c_tensor, idx in zip(i_tensors, v, range(len(i_tensors))):
                    if torch.is_tensor(i_tensor):
                        close = check_tensor_close(i_tensor=i_tensor, c_tensor=c_tensor)
                        if close:
                            test_coverage = test_coverage + 1
                            logger.info(f"Pass {k}'s No.{idx} Tensor Check.")
                        else:
                            logger.error(f"Failure {k}'s No.{idx} Tensor Check!")
        test_coverage = test_coverage / self.check_tensors_num()
        logger.info(f"Test Coverage is: {test_coverage:.2f}%.")
        return all_close

    def check_tensors_num(self):
        len = 0
        for key, values in TorchHook.results_dict.items():
            for v in values:
                len = len + 1
        return len

    @staticmethod
    def resert_results_dict():
        TorchHook.results_dict = dict()