'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:16:36
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:29:55
Description  : file content
FilePath     : /SVTAS/svtas/profiling/serving_profiler.py
'''
from .base_profiler import BaseProfiler
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('profiler')
class ServingPerformanceProfiler(BaseProfiler):
    def __init__(self, profile_step: int = 1) -> None:
        super().__init__(profile_step)
    
    def init_profiler(self, model, *args, **kwargs):
        return super().init_profiler(model, *args, **kwargs)
    
    def shutdown_profiler(self):
        return super().shutdown_profiler()
    
    def start_profile(self) -> None:
        return super().start_profile()
    
    def end_profile(self) -> None:
        return super().end_profile()
    
    def print_model_profile(self) -> None:
        return super().print_model_profile()