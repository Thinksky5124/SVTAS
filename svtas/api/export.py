'''
Author       : Thyssen Wen
Date         : 2023-10-19 15:48:36
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 14:10:51
Description  : file content
FilePath     : /SVTAS/svtas/api/export.py
'''
import torch
import torch.profiler
from ..utils.logger import get_logger

from svtas.engine import BaseEngine
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import BaseModelPipline

@torch.no_grad()
def export(local_rank,
            nprocs,
            cfg,
            args):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    
    # 1. Construct model.
    model_pipline: BaseModelPipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # 2. Construct dataset and dataloader.
    fake_dataloader = AbstractBuildFactory.create_factory('dataloader').create(cfg.DATALOADER)
    
     # 3. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    profile_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    profile_engine.set_dataloader(fake_dataloader)
    profile_engine.running_mode = 'export'
    
    # 6. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        profile_engine.resume()
        
    # 7. run engine
    profile_engine.init_engine()
    profile_engine.run()
    profile_engine.shutdown()