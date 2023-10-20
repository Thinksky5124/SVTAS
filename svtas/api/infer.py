'''
Author       : Thyssen Wen
Date         : 2022-09-23 20:51:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 17:02:23
Description  : infer script api
FilePath     : /SVTAS/svtas/api/infer.py
'''
from svtas.utils.logger import get_logger
from svtas.utils import mkdir
from svtas.utils import AbstractBuildFactory
from svtas.engine import BaseEngine
from ..utils.collect_env import collect_env

def infer(local_rank,
          nprocs,
          cfg,
          args):
    """
    Infer model entry
    """
    logger = get_logger("SVTAS")
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # 2. build metirc
    metric_cfg = cfg.METRIC
    metrics = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = False
        metrics[k] = AbstractBuildFactory.create_factory('metric').create(v)
    
    # 3. construct model_pipline
    model_pipline = AbstractBuildFactory.create_factory('model_pipline').create(cfg.MODEL_PIPLINE)

    # 4. Construct Dataset
    batch_size = cfg.DATALOADER.get('batch_size', 8)
    infer_pipeline = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.DATASETPIPLINE.infer)
    infer_dataset_config = cfg.DATASET.infer
    infer_dataset_config['pipeline'] = infer_pipeline
    infer_dataset_config['batch_size'] = batch_size * nprocs
    infer_dataset_config['local_rank'] = local_rank
    infer_dataset_config['nprocs'] = nprocs
    infer_dataloader_config = cfg.DATALOADER
    infer_dataloader_config['dataset'] = AbstractBuildFactory.create_factory('dataset').create(infer_dataset_config)
    infer_dataloader_config['collate_fn'] = AbstractBuildFactory.create_factory('dataset_pipline').create(cfg.COLLATE.infer)
    infer_dataloader = AbstractBuildFactory.create_factory('dataloader').create(infer_dataloader_config)
    
    # 5. build engine
    engine_config = cfg.ENGINE
    engine_config['logger_dict'] = cfg.LOGGER_LIST
    engine_config['metric'] = metrics
    engine_config['model_pipline'] = model_pipline
    engine_config['model_name'] = model_name
    infer_engine: BaseEngine = AbstractBuildFactory.create_factory('engine').create(engine_config)
    infer_engine.set_dataloader(infer_dataloader)
    infer_engine.running_mode = 'infer'
    
    # 6. resume engine
    if cfg.ENGINE.checkpointor.get('load_path', None) is not None:
        infer_engine.resume()
        
    # 7. run engine
    infer_engine.init_engine()
    infer_engine.run()
    infer_engine.shutdown()