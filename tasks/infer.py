'''
Author       : Thyssen Wen
Date         : 2022-09-23 20:51:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-09-23 21:04:42
Description  : infer script api
FilePath     : /ETESVS/tasks/infer.py
'''
import torch
from utils.logger import get_logger
from .runner import Runner
from utils.recorder import build_recod
import time
import numpy as np

import model.builder as model_builder
import loader.builder as dataset_builder
import metric.builder as metric_builder
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import clever_format
from utils.collect_env import collect_env

def infer(cfg,
          args,
          local_rank,
          nprocs,
          weights=None,
          validate=True,):
    """
    Infer model entry
    """
    logger = get_logger("SVTAS")
    
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    model_name = cfg.model_name

    # 1. Construct model.
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.construct model
        model = model_builder.build_model(cfg.MODEL).cuda()
        criterion = model_builder.build_loss(cfg.MODEL.loss).cuda()

    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = model_builder.build_model(cfg.MODEL).cuda(local_rank)
        criterion = model_builder.build_loss(cfg.MODEL.loss).cuda()
    
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # wheather batch train
    batch_train = False
    if cfg.COLLATE.name in ["BatchCompose"]:
        batch_train = True

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE)
    test_Pipeline = dataset_builder.build_pipline(cfg.PIPELINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_Pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader = torch.utils.data.DataLoader(
        dataset_builder.build_dataset(test_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)

    if local_rank < 0:
        checkpoint = torch.load(weights)
    else:
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(weights, map_location=map_location)

    state_dicts = checkpoint['model_state_dict']

    if nprocs > 1:
        model.module.load_state_dict(state_dicts)
    else:
        model.load_state_dict(state_dicts)


    # add params to metrics
    Metric = metric_builder.build_metric(cfg.METRIC)
    
    record_dict = build_recod(cfg.MODEL.architecture, mode="validation")

    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = Runner(logger=logger,
                video_batch_size=video_batch_size,
                Metric=Metric,
                record_dict=record_dict,
                cfg=cfg,
                model=model,
                criterion=criterion,
                post_processing=post_processing,
                nprocs=nprocs,
                local_rank=local_rank,
                runner_mode='test')

    runner.epoch_init()
    r_tic = time.time()
    for i, data in enumerate(test_dataloader):
        if batch_train is True:
            runner.run_one_batch(data=data, r_tic=r_tic)
        else:
            runner.run_one_iter(data=data, r_tic=r_tic)
        r_tic = time.time()

    logger.info(f'infering {model_name} finished')