'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 14:34:15
Description: train script api
FilePath     : /SVTAS/svtas/tasks/train.py
'''

import torch
from svtas.utils.logger import get_logger
from svtas.utils.save_load import mkdir
from svtas.utils import AbstractBuildFactory
from svtas.engine import BaseEngine

def train(cfg,
          args,
          local_rank,
          nprocs):
    """Train model entry
    """
    
    # 1. init logger and output folder
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    criterion_metric_name = cfg.get("criterion_metric_name", "F1@0.50")
    mkdir(output_dir)
    
    # 2. build metirc
    metric_cfg = cfg.METRIC
    metrics = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = True
        metrics[k] = AbstractBuildFactory.create_factory('metric').create(v)

    # 3. construct Pipeline
    train_Pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.PIPELINE.train)
    val_Pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.PIPELINE.test)

    # wheather batch train
    batch_train = False
    if cfg.COLLATE.train.name in ["BatchCompose"]:
        batch_train = True
    batch_test = False
    if cfg.COLLATE.test.name in ["BatchCompose"]:
        batch_test = True
        
    # 4. Construct Dataset
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    num_workers = cfg.DATASET.get('num_workers', 0)
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_Pipeline
    train_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    train_dataset_config['video_batch_size'] = video_batch_size * nprocs
    train_dataset_config['local_rank'] = local_rank
    train_dataset_config['nprocs'] = nprocs
    train_dataloader = torch.utils.data.DataLoader(
        AbstractBuildFactory.create_factory('dataset').create(train_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.train))        

    # 5. build model_pipline
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # 6. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['metric'] = metrics
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    train_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    train_engine.set_dataloader(train_dataloader)
    train_engine.running_mode = 'train'

    # 7. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        train_engine.resume()
    
    if cfg.ENGINE.iter_method.get("test_interval", -1) > -1:
        val_dataset_config = cfg.DATASET.test
        val_dataset_config['pipeline'] = val_Pipeline
        val_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
        val_dataset_config['video_batch_size'] = video_batch_size * nprocs
        val_dataset_config['local_rank'] = local_rank
        val_dataset_config['nprocs'] = nprocs
        val_dataloader = torch.utils.data.DataLoader(
            AbstractBuildFactory.create_factory('dataset').create(val_dataset_config),
            batch_size=temporal_clip_batch_size,
            num_workers=num_workers,
            collate_fn=AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test))
        
        engine_config = cfg.ENGINE
        engine_config['logger_dict'] = cfg.LOGGER_LIST
        engine_config['metric'] = metrics
        engine_config['model_pipline'] = model_pipline
        engine_config['model_name'] = model_name
        test_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
        test_engine.set_dataloader(val_dataloader)
        test_engine.running_mode = 'validation'

        def test_run(best_score: float) -> float:
            test_engine.init_engine()
            test_engine.iter_method.memory_score = best_score
            test_engine.run()
            return test_engine.iter_method.best_score

        train_engine.iter_method.register_test_hook(test_run)
        
    # 8. train
    train_engine.init_engine()
    train_engine.run()

    # 9. close
    train_engine.shutdown()
    test_engine.shutdown()