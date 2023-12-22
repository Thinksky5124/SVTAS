'''
Author       : Thyssen Wen
Date         : 2023-12-19 14:52:39
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-19 20:51:44
Description  : Numerical Range Profiler Class
FilePath     : /SVTAS/svtas/profiling/numerical_range_profiler.py
'''
import os
import numpy as np
import pandas as pd
from functools import partial
from typing import Dict, List, Any
from .base_profiler import BaseProfiler
from svtas.utils import AbstractBuildFactory, is_seaborn_available, is_torch_available
from svtas.utils.logger import get_root_logger_instance

if is_torch_available():
    import torch

@AbstractBuildFactory.register('profiler')
class NumericalRangeProfiler(BaseProfiler):
    RESULT_DICT: Dict = None

    def __init__(self,
                 profile_step: int = 1,
                 profile_layers: List[Any] = ['all'],
                 sample_elem_num: int = 500,
                 profile_activation: bool = True,
                 profile_weight: bool = True,
                 need_plot: bool = True,
                 plot_path: str = "./output") -> None:
        super().__init__(profile_step)
        self.profile_layers = profile_layers
        self.profile_activation = profile_activation
        self.profile_weight = profile_weight
        self.need_plot = need_plot
        self.plot_path = plot_path
        self.sample_elem_num = sample_elem_num

        self.profile_all_layers = False
        if len(self.profile_layers) > 0 and self.profile_layers[0] == 'all':
            self.profile_all_layers = True
    
    def init_profiler(self, model):
        self.model = model
    
    def shutdown_profiler(self):
        return super().shutdown_profiler()
    
    def start_profile(self) -> None:
        logger = get_root_logger_instance()
        logger.info("Numerical range profiler started")
        self.RESULT_DICT = dict()

        def post_hook(module, input, output, name: str=''):
            if name in self.RESULT_DICT:
                if torch.is_tensor(output):
                    self.RESULT_DICT[name] += output.detach().cpu().numpy()
                else:
                    self.RESULT_DICT[name] += output[0].detach().cpu().numpy()
            else:
                if torch.is_tensor(output):
                    self.RESULT_DICT[name] = output.detach().cpu().numpy()
                else:
                    self.RESULT_DICT[name] = output[0].detach().cpu().numpy()

        for name, module in self.model.named_modules():
            if self.profile_all_layers:
                module.__numerical_range_handle__ = module.register_forward_hook(partial(post_hook, name=name))
            else:
                if name in self.profile_layers or type(module) in self.profile_layers:
                    module.__numerical_range_handle__ = module.register_forward_hook(partial(post_hook, name=name))
    
    def end_profile(self) -> None:
        def remove_profile_attrs(module):
            if hasattr(module, "__numerical_range_handle__"):
                del module.__numerical_range_handle__

        self.model.apply(remove_profile_attrs)
        logger = get_root_logger_instance()
        logger.info("Numerical range profiler finished")
    
    def print_model_profile(self, profile_step=1) -> None:
        logger = get_root_logger_instance()
        logger.log("\n-------------------------- SVTAS Numerical Range Profiler --------------------------")
        logger.log(f'Profile Summary at step {profile_step}:')
        logger.log(f'Layer numerical range:')

        if profile_step > 1:
            for key, value in self.RESULT_DICT.items():
                self.RESULT_DICT[key] /= profile_step
                self.RESULT_DICT[key] = self.RESULT_DICT[key].flatten()

        for key, value in self.RESULT_DICT.items():
            max_n = np.max(value)
            min_n = np.min(value)
            median_n = np.median(value)
            logger.log(f'{key} - max: {max_n}, min: {min_n}, median: {median_n};')
        
        if self.need_plot:
            if is_seaborn_available():
                import seaborn as sns
            else:
                raise ImportError("Must install seaborn!")
            
            pd_dict = {'layer_name': [], 'data': []}
            for key, value in self.RESULT_DICT.items():
                pd_dict['layer_name'] += [key] * self.sample_elem_num
                pd_dict['data'] += list(np.random.choice(value, self.sample_elem_num, replace=False))
            
            array_data = pd.DataFrame.from_dict(pd_dict)
            ax = sns.boxplot(array_data, x='layer_name', y='data')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.figure.savefig(os.path.join(self.plot_path, 'numerical_range.png'), bbox_inches='tight') 
