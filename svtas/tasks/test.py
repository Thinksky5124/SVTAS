'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-22 15:00:14
Description: test script api
FilePath     : /SVTAS/svtas/tasks/test.py
'''
import torch
from ..utils.logger import get_logger
from ..runner.runner import Runner
from ..utils.recorder import build_recod
import time
import numpy as np

from ..model.builder import build_model
from ..model.builder import build_loss
from ..loader.builder import build_dataset
from ..loader.builder import build_pipline
from ..metric.builder import build_metric
from ..model.builder import build_post_precessing
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import clever_format
from ..utils.collect_env import collect_env
import warnings
try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP
except:
    pass

@torch.no_grad()
def test(cfg,
         args,
         local_rank,
         nprocs,
         use_amp=False,
         weights=None):
    logger = get_logger("SVTAS")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("SVTAS", tensorboard=args.use_tensorboard)
    # wheather use amp
    if use_amp is True:
        logger.info("use amp")
    need_grad_accumulate = cfg.OPTIMIZER.get('need_grad_accumulate', True)
    
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # 1. Construct model.
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.construct model
        model = build_model(cfg.MODEL).to(device)
        criterion = build_loss(cfg.MODEL.loss).to(device)

        # wheather to use amp
        if use_amp is True:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = build_model(cfg.MODEL).cuda(local_rank)
        criterion = build_loss(cfg.MODEL.loss).cuda(local_rank)

        device = torch.device('cuda:{}'.format(local_rank))
    
        # wheather to use amp
        if use_amp is True:
            model = convert_syncbn_model(model).to(device)
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # wheather batch train
    batch_test = False
    if cfg.COLLATE.test.name in ["BatchCompose"]:
        batch_test = True

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = build_pipline(cfg.COLLATE.test)
    test_Pipeline = build_pipline(cfg.PIPELINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_Pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader = torch.utils.data.DataLoader(
        build_dataset(test_dataset_config),
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

    if use_amp is True:
        amp.load_state_dict(checkpoint['amp'])

    # add params to metrics
    metric_cfg = cfg.METRIC
    Metric = dict()
    for k, v in metric_cfg.items():
        Metric[k] = build_metric(v)
    
    record_dict = build_recod(cfg.MODEL.architecture, mode="validation")

    post_processing = build_post_precessing(cfg.POSTPRECESSING)

    runner = Runner(logger=logger,
                video_batch_size=video_batch_size,
                Metric=Metric,
                record_dict=record_dict,
                cfg=cfg,
                model=model,
                criterion=criterion,
                post_processing=post_processing,
                use_amp=use_amp,
                nprocs=nprocs,
                local_rank=local_rank,
                runner_mode='test',
                need_grad_accumulate=need_grad_accumulate)

    runner.epoch_init()
    r_tic = time.time()
    for i, data in enumerate(test_dataloader):
        if batch_test is True:
            runner.run_one_batch(data=data, r_tic=r_tic)
        elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
            runner.run_one_iter(data=data, r_tic=r_tic)
        else:
            break
        r_tic = time.time()

    if local_rank <= 0:
        # metric output
        for k, v in runner.Metric.items():
            v.accumulate()