'''
Author       : Thyssen Wen
Date         : 2023-10-18 17:06:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:28:59
Description  : file content
FilePath     : /SVTAS/svtas/profiling/flops_param_profiler/fvcore_profiler.py
'''
from typing import Dict
from svtas.utils import is_fvcore_available
from svtas.utils.logger import BaseLogger
from ..base_profiler import BaseProfiler
from svtas.utils.logger import get_root_logger_instance
from svtas.utils import AbstractBuildFactory

if is_fvcore_available():
    from fvcore.nn import FlopCountAnalysis, flop_count_table

@AbstractBuildFactory.register('profiler')
class FvcoreProfiler(BaseProfiler):
    def __init__(self, profile_step: int = 1) -> None:
        super().__init__(profile_step)
    
    def init_profiler(self, model, *args, **kwargs):
        self.model = model
    
    def shutdown_profiler(self):
        return super().shutdown_profiler()
    
    def start_profile(self) -> None:
        logger = get_root_logger_instance()
        logger.info("="*20)
        # if computing the flops of the functionals in a module
        def pre_hook(module, input):
            logger.info('Use fvcore FlopCountAnalysis function')
            flops = FlopCountAnalysis(self.model_pipline.model, (input))
            self.info = flops

        if not hasattr(self.model, "__pre_hook_handle__"):
            self.model.__pre_hook_handle__ = self.model.register_forward_pre_hook(pre_hook)
    
    def end_profile(self) -> None:
        if hasattr(self.model, "__pre_hook_handle__"):
            self.model.__pre_hook_handle__.remove()
            del self.model.__pre_hook_handle__
    
    def print_model_profile(self) -> None:
        logger = get_root_logger_instance()
        logger.info("flop_count_table: \n" + flop_count_table(self.info))
        logger.info("="*20)