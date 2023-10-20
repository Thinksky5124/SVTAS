'''
Author       : Thyssen Wen
Date         : 2023-10-08 16:20:46
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 20:04:06
Description  : file content
FilePath     : /SVTAS/svtas/tasks/visualize_loss.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
from svtas.utils import mkdir
from svtas.utils.logger import get_logger, setup_logger, coloring
from svtas.utils import AbstractBuildFactory

def visulize_loss(local_rank,
                  nprocs,
                  cfg,
                  args):
    logger = get_logger("SVTAS")
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # 2. build metirc
    model_name = cfg.model_name
    metric_cfg = cfg.METRIC
    metrics = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = True
        metrics[k] = AbstractBuildFactory.create_factory('metric').create(v)

    # 3. construct Pipeline
    train_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE.train)
        
    # 4. Construct Dataset
    batch_size = cfg.DATALOADER.get('batch_size', 8)
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_pipeline
    train_dataset_config['batch_size'] = batch_size * nprocs
    train_dataset_config['local_rank'] = local_rank
    train_dataset_config['nprocs'] = nprocs
    train_dataloader_config = cfg.DATALOADER
    train_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(train_dataset_config)
    train_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.train)
    train_dataloader = AbstractBuildFactory.create_factory('dataloader').create(train_dataloader_config)     

    val_Pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE.test)
    val_dataset_config = cfg.DATASET.test
    val_dataset_config['pipeline'] = val_Pipeline
    val_dataset_config['batch_size'] = batch_size * nprocs
    val_dataset_config['local_rank'] = local_rank
    val_dataset_config['nprocs'] = nprocs
    val_dataloader_config = cfg.DATALOADER
    val_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(val_dataset_config)
    val_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.test)
    val_dataloader = AbstractBuildFactory.create_factory('dataloader').create(val_dataloader_config)
    
    # 5. build model_pipline
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # 6. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['metric'] = metrics
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    visual_engine = AbstractBuildFactory.create_factory('engine').create(engine_config)

    # 7. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        visual_engine.resume()
    visual_engine.set_all_dataloader(train_dataloader, val_dataloader)
    visual_engine.init_engine()
    visual_engine.run()
    visual_engine.shutdown()
    logger.info(coloring("Finish plot loss landspace!"))