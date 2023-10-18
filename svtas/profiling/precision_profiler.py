'''
Author       : Thyssen Wen
Date         : 2023-10-18 19:46:46
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:29:37
Description  : file content
FilePath     : /SVTAS/svtas/profiling/precision_profiler.py
'''
from typing import Dict
from .base_profiler import BaseProfiler
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('profiler')
class PrecisionCompareProfiler(BaseProfiler):
    def __init__(self,
                 profile_step: int = 1,
                 comapre_pair_name: Dict[str, str] = None) -> None:
        super().__init__(profile_step)
        self.comapre_pair_name = comapre_pair_name
    
    def init_profiler(self, model):
        self.criterion_model = model
    
    def shutdown_profiler(self):
        return super().shutdown_profiler()
    
    def start_profile(self) -> None:
        return super().start_profile()
    
    def end_profile(self) -> None:
        return super().end_profile()
    
    def print_model_profile(self) -> None:
        return super().print_model_profile()