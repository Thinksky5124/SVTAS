'''
Author       : Thyssen Wen
Date         : 2022-11-21 10:53:11
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 21:56:14
Description  : file content
FilePath     : /SVTAS/svtas/optimizer/grad_utils.py
'''
from ..utils import AbstractBuildFactory
import torch

@AbstractBuildFactory.register('optimizer')
class GradClip(object):
    def __init__(self,
                 max_norm=40,
                 norm_type=2,
                 **kwargs) -> None:
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, param):
        torch.nn.utils.clip_grad_norm_(parameters=param, max_norm=self.max_norm, norm_type=self.norm_type)

@AbstractBuildFactory.register('optimizer')
class GradAccumulate(object):
    """Grad Accumulate Class
    It support accumulate grad from `iter` and `conf`.
    
    1. When use `iter`, you must call `update` before every optimizer step to judge wheater to update param.

    2. When use `conf`, you should call `set_update_conf` to update param.
    """
    def __init__(self,
                 accumulate_type: str,
                 accumulate_step: int = 1) -> None:
        assert accumulate_type in ['iter', 'conf'], f"Unsupport accumulate_type: {accumulate_type}!"
        self.accumulate_type = accumulate_type
        self.accumulate_step = accumulate_step
        self._readly_to_update_param_flag = False
        self._cnt_iter_num = 0
    
    def judge_iter(self) -> bool:
        self._cnt_iter_num += 1
        if self._cnt_iter_num == self.accumulate_step:
            self._cnt_iter_num = 0
            return True
        else:
            return False
    
    def judge_conf(self) -> bool:
        if self._readly_to_update_param_flag:
            self._readly_to_update_param_flag = False
            return True
        else:
            return False
        
    @property
    def update(self) -> bool:
        if self.accumulate_type == 'conf':
            return self.judge_conf()
        elif self.accumulate_type == 'iter':
            return self.judge_iter()
    
    def set_update_conf(self):
        self._readly_to_update_param_flag = True
