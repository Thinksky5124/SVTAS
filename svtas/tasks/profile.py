'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 14:58:38
Description: test script api
FilePath     : /SVTAS/svtas/tasks/profile.py
'''
import torch
import torch.profiler
from ..utils.logger import get_logger
import time
import numpy as np
import datetime

from svtas.engine import BaseEngine
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import BaseModelPipline
from ..utils.collect_env import collect_env

@torch.no_grad()
def profile(local_rank,
            nprocs,
            cfg,
            args):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # 1. Construct model.
    model_pipline: BaseModelPipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)
    device = model_pipline.device

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    batch_size = cfg.DATALOADER.get('batch_size', 8)
    test_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_pipeline
    test_dataset_config['batch_size'] = batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader_config = cfg.DATALOADER
    test_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(test_dataset_config)
    test_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test)
    test_dataloader = AbstractBuildFactory.create_factory('dataloader').create(test_dataloader_config)
    
     # 3. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    profile_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    profile_engine.set_dataloader(test_dataloader)
    profile_engine.running_mode = 'test'
    
    # 6. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        profile_engine.resume()
        
    # 7. run engine
    profile_engine.init_engine()
    profile_engine.run()
    profile_engine.shutdown()