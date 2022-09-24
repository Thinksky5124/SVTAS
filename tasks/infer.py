'''
Author       : Thyssen Wen
Date         : 2022-09-23 20:51:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-09-24 13:15:33
Description  : infer script api
FilePath     : \ETESVS\tasks\infer.py
'''
import torch
from utils.logger import get_logger
from .runner import Runner
from utils.logger import AverageMeter
import time
import numpy as np
import onnx
import onnxruntime
import os

import model.builder as model_builder
import loader.builder as dataset_builder
import metric.builder as metric_builder
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
        model = model_builder.build_model(cfg.MODEL).to(device)
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = model_builder.build_model(cfg.MODEL).cuda(local_rank)
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
    
    # Export Model
    # export path construct
    export_path = os.path.join("output", cfg.model_name, cfg.model_name + ".onnx")

    if os.path.exists(os.path.join("output", cfg.model_name)) is False:
        os.mkdir(os.path.join("output", cfg.model_name))
        
    # model param flops caculate
    if cfg.MODEL.architecture not in ["FeatureSegmentation"]:
        x_shape = [cfg.DATASET.test.clip_seg_num, 3, cfg.PIPELINE.test.transform.transform_list.Resize[0], cfg.PIPELINE.test.transform.transform_list.Resize[1]]
        mask_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        labels_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).cuda()
            mask = torch.randn([optimal_batch_size] + mask_shape).cuda()
            label = torch.ones([optimal_batch_size] + labels_shape).cuda()
            return dict(input_data=dict(imgs=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)
    else:
        x_shape = [cfg.DATASET.test.clip_seg_num, 2048]
        mask_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        labels_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).cuda()
            mask = torch.randn([optimal_batch_size] + mask_shape).cuda()
            label = torch.ones([optimal_batch_size] + labels_shape).cuda()
            return dict(input_data=dict(feature=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)

    logger.info("Start exporting ONNX model!")
    torch.onnx.export(
        model,
        dummy_input['input_data'],
        export_path,
        opset_version=11,
        input_names=['input_data', 'masks'],
        output_names=['output'])
    logger.info("Finish exporting ONNX model to " + export_path + " !")
    ort_session = onnxruntime.InferenceSession(export_path)

    # add params to metrics
    Metric = metric_builder.build_metric(cfg.METRIC)
    
    record_dict = {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f')}

    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = Runner(logger=logger,
                video_batch_size=video_batch_size,
                Metric=Metric,
                record_dict=record_dict,
                cfg=cfg,
                model=ort_session,
                post_processing=post_processing,
                nprocs=nprocs,
                local_rank=local_rank)

    runner.epoch_init()
    r_tic = time.time()
    for i, data in enumerate(test_dataloader):
        if batch_train is True:
            runner.run_one_batch(data=data, r_tic=r_tic)
        else:
            runner.run_one_iter(data=data, r_tic=r_tic)
        r_tic = time.time()

    logger.info(f'infering {model_name} finished')