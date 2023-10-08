'''
Author       : Thyssen Wen
Date         : 2022-05-17 16:58:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 13:52:06
Description  : Extract video feature script
FilePath     : /SVTAS/svtas/tasks/extract.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
from svtas.utils.logger import get_logger
from svtas.utils import AbstractBuildFactory

def extract(cfg,
            args,
            local_rank,
            nprocs):
    logger = get_logger("SVTAS")
    model_name = cfg.model_name

    if cfg.ENGINE.name == "ExtractOpticalFlowEngine":
        video_batch_size = cfg.DATASET.get('video_batch_size', 1)
        assert video_batch_size == 1, "Only support 1 batch size"

    # construct model
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # construct dataloader
    temporal_clip_batch_size = cfg.DATALOADER.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATALOADER.get('video_batch_size', 8)
    test_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE)
    test_dataset_config = cfg.DATASET.config
    test_dataset_config['pipeline'] = test_pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size
    test_dataloader_config = cfg.DATALOADER
    test_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(test_dataset_config)
    test_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test)
    test_dataloader = AbstractBuildFactory.create_factory('dataloader').create(test_dataloader_config)
    
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name + "_extract"
    extract_engine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    extract_engine.set_dataloader(test_dataloader)

    extract_engine.init_engine()
    extract_engine.run()
    logger.info("Finish all extracting!")
    extract_engine.shutdown()
