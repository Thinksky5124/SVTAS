'''
Author       : Thyssen Wen
Date         : 2023-10-18 17:06:29
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:19:35
Description  : file content
FilePath     : /SVTAS/svtas/profiling/flops_param_profiler/mmcv_profiler.py
'''
from typing import Dict
from svtas.utils import is_mmcv_available
from svtas.utils.logger import BaseLogger
from ..base_profiler import BaseProfiler
from svtas.utils.misc import clever_format
from svtas.utils.logger import get_root_logger_instance
from svtas.utils import AbstractBuildFactory

if is_mmcv_available():
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info

@AbstractBuildFactory.register('profiler')
class MMCVProfiler(BaseProfiler):
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
            logger.info('Use mmcv get_model_complexity_info function')

            def input_constructor(input_shape):
                return input
            
            flops_number, params_number = get_model_complexity_info(self.model,
                                                                    input_shape=(),
                                                                    input_constructor=input_constructor,
                                                                    print_per_layer_stat=False,
                                                                    as_strings=False)
            self.info = [flops_number, params_number]

        if not hasattr(self.model, "__pre_hook_handle__"):
            self.model.__pre_hook_handle__ = self.model.register_forward_pre_hook(pre_hook)
    
    def end_profile(self) -> None:
        if hasattr(self.model, "__pre_hook_handle__"):
            self.model.__pre_hook_handle__.remove()
            del self.model.__pre_hook_handle__
    
    def print_model_profile(self) -> None:
        logger = get_root_logger_instance()
        flops, params = clever_format(self.info, "%.6f")
        logger.info("Total FLOPs:" + flops + ", Total params:" + params)
        logger.info("="*20)