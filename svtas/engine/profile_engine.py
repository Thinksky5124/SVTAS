'''
Author       : Thyssen Wen
Date         : 2023-10-11 14:36:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-12 16:14:23
Description  : file content
FilePath     : /SVTAS/svtas/engine/profile_engine.py
'''
import os
from typing import Dict, List
import torch
import numpy as np

from svtas.utils.logger import get_root_logger_instance
from .standalone_engine import StandaloneEngine
from svtas.utils import AbstractBuildFactory
from svtas.profiling import BaseProfiler

@AbstractBuildFactory.register('engine')
class TorchStandaloneProfilerEngine(StandaloneEngine):
    extra_profiler: Dict[str, BaseProfiler]

    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 metric: Dict = {},
                 torch_profile_cfg: Dict = None,
                 extra_profiler: Dict[str, Dict] = {},
                 running_mode='profile') -> None:
        super().__init__(model_name, model_pipline, logger_dict, record,
                         metric, iter_method, checkpointor, running_mode)
        full_torch_profile_cfg = dict(
            wait = 1,  warmup = 1, active = 3, repeat = 2, record_shapes = True, profile_memory = True, with_stack = True
        )
        self.torch_profile = False
        if torch_profile_cfg is not None:
            for key, value in torch_profile_cfg.items():
                full_torch_profile_cfg[key] = value
            self.full_torch_profile_cfg = full_torch_profile_cfg
            self.torch_profile = True
        else:
            self.full_torch_profile_cfg = full_torch_profile_cfg
        self.extra_profiler = {}
        for key, cfg in extra_profiler.items():
            self.extra_profiler[key] = AbstractBuildFactory.create_factory('profiler').create(cfg)
    
    def run(self):
        self.iter_method.save_interval = 1
        # model param flops caculate

        if self.torch_profile: 
            self.model_pipline.train()
            with torch.set_grad_enabled(True):
                with torch.profiler.profile(
                    schedule=torch.profiler.schedule(wait=self.full_torch_profile_cfg['wait'], warmup=self.full_torch_profile_cfg['warmup'],
                                                    active=self.full_torch_profile_cfg['active'], repeat=self.full_torch_profile_cfg['repeat']),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.checkpointor.save_path, self.model_name)),
                    record_shapes=self.full_torch_profile_cfg['record_shapes'],
                    profile_memory=self.full_torch_profile_cfg['profile_memory'],
                    with_stack=self.full_torch_profile_cfg['with_stack']
                    ) as prof:
                        for iter_cnt in self.iter_method.run():
                            prof.step()
            self.model_pipline.eval()
        
        if len(self.extra_profiler) > 0:
            logger = get_root_logger_instance()
            for key, prof in self.extra_profiler.items():
                logger.log(f"Start {key} Profiling ......")
                prof.init_profiler(self.model_pipline.model)
                init_flag = False
                for iter_cnt in self.iter_method.run():
                    if iter_cnt > self.full_torch_profile_cfg['warmup'] and not init_flag:
                        prof.start_profile()
                        init_flag = True
                    elif iter_cnt > self.full_torch_profile_cfg['warmup'] + prof.profile_step:
                        prof.print_model_profile(profile_step=iter_cnt)
                        prof.end_profile()
                        break
                prof.shutdown_profiler()
                logger.log(f"End {key} Profiling.")
