'''
Author       : Thyssen Wen
Date         : 2022-09-23 20:51:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 11:51:02
Description  : infer script api
FilePath     : /SVTAS/svtas/tasks/infer.py
'''
import torch
from ..utils.logger import get_logger
from ..engine.infer_engine import InferONNXEngine 
from ..utils.logger import AverageMeter
from .debug_infer_forward_func import infer_forward, debugger
import time
import numpy as np
import onnx
import onnxruntime
import os
from types import MethodType

from ..utils.save_load import mkdir
# from ..model.builder import build_model
# from ..loader.builder import build_dataset
# from ..loader.builder import build_pipline
# from ..metric.builder import build_metric
# from ..model.builder import build_post_precessing
from ..utils.collect_env import collect_env

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
    debugger.logger = logger

    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    model_name = cfg.model_name

    if cfg.INFER.device is None:
        device = torch.device("cpu")
    else:
        device = torch.device(cfg.INFER.device)

    # 1. Construct model.
    if local_rank < 0:
        # 1.construct model
        model = build_model(cfg.MODEL).to(device)

    # wheather batch train
    batch_infer = False
    if cfg.COLLATE.infer.name in ["BatchCompose"]:
        batch_infer = True

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    infer_num_workers = cfg.DATASET.get('infer_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    assert video_batch_size == 1, "Only Support video_batch_size equal to 1!"
    sliding_concate_fn = build_pipline(cfg.COLLATE.infer)
    infer_Pipeline = build_pipline(cfg.DATASETPIPLINE.infer)
    infer_dataset_config = cfg.DATASET.infer
    infer_dataset_config['pipeline'] = infer_Pipeline
    infer_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    infer_dataset_config['video_batch_size'] = video_batch_size * nprocs
    infer_dataset_config['local_rank'] = local_rank
    infer_dataset_config['nprocs'] = nprocs
    infer_dataloader = torch.utils.data.DataLoader(
        build_dataset(infer_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=infer_num_workers,
        collate_fn=sliding_concate_fn)

    if local_rank < 0:
        checkpoint = torch.load(weights, map_location=device)

    state_dicts = checkpoint['model_state_dict']

    if nprocs > 1:
        model.module.load_state_dict(state_dicts)
    else:
        model.load_state_dict(state_dicts)
    
    # Export Model
    # export path construct
    export_path = os.path.join("output", cfg.model_name, cfg.model_name + ".onnx")

    if os.path.exists(os.path.join("output", cfg.model_name)) is False:
        mkdir(os.path.join("output", cfg.model_name))
        
    # model param flops caculate
    if cfg.MODEL.architecture not in ["FeatureSegmentation"]:
        for transform_op in list(cfg.DATASETPIPLINE.test.transform.transform_dict.values())[0]:
            if list(transform_op.keys())[0] in ['CenterCrop']:
                image_size = transform_op['CenterCrop']['size']
        x_shape = [cfg.DATASET.infer.clip_seg_num, 3, image_size, image_size]
        mask_shape = [cfg.DATASET.infer.clip_seg_num * cfg.DATASET.infer.sample_rate]
        input_shape = (x_shape, mask_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).to(device)
            mask = torch.randn([optimal_batch_size] + mask_shape).to(device)
            return dict(input_data=dict(imgs=x, masks=mask))
        dummy_input = input_constructor(input_shape)
    else:
        x_shape = [cfg.DATASET.infer.clip_seg_num, 2048]
        mask_shape = [cfg.DATASET.infer.clip_seg_num * cfg.DATASET.infer.sample_rate]
        input_shape = (x_shape, mask_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).to(device)
            mask = torch.randn([optimal_batch_size] + mask_shape).to(device)
            return dict(input_data=dict(feature=x, masks=mask))
        dummy_input = input_constructor(input_shape)

    # add params to metrics
    metric_cfg = cfg.METRIC
    Metric = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = True
        Metric[k] = build_metric(v)
    
    record_dict = {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f')}

    post_processing = build_post_precessing(cfg.POSTPRECESSING)

    if cfg.INFER.infer_engine.name == "onnx":
        if validate is True:
            logger.info(f'debuging {model_name} weather precision align ...')
            model.forward = MethodType(infer_forward, model)

        logger.info("Start exporting ONNX model!")
        torch.onnx.export(
            model,
            dummy_input,
            export_path,
            opset_version=cfg.INFER.infer_engine.opset_version,
            input_names=cfg.INFER.input_names,
            output_names=cfg.INFER.output_names)
        logger.info("Finish exporting ONNX model to " + export_path + " !")

        # Debug Model
        if validate is True:
            logger.info(f'debuging {model_name} weather precision align ...')
            export_debug_path = os.path.join("output", cfg.model_name, cfg.model_name + "_debug.onnx")
            # Extrator Statisc Model
            debugger.extract_debug_model(export_path, export_debug_path)

            # convert dummpy_input tensor to ndarray
            input_data = {}
            for key, value in dummy_input['input_data'].items():
                input_data[key] = value.numpy()
            debugger.run_debug_model(input_data, export_debug_path)
            debugger.print_debug_result()
            logger.info(f'debuging {model_name} finished !')
            
            return
        else:
            ort_session = onnxruntime.InferenceSession(export_path)
            runner = InferONNXRunner(logger=logger,
                        video_batch_size=video_batch_size,
                        Metric=Metric,
                        record_dict=record_dict,
                        cfg=cfg,
                        model=ort_session,
                        post_processing=post_processing,
                        nprocs=nprocs,
                        local_rank=local_rank)
                    
    elif cfg.INFER.infer_engine.name == "TensorRT":
        raise NotImplementedError
    else:
        raise NotImplementedError

    runner.epoch_init()
    r_tic = time.time()
    for i, data in enumerate(infer_dataloader):
        if batch_infer is True:
            runner.run_one_batch(data=data, r_tic=r_tic)
        else:
            runner.run_one_iter(data=data, r_tic=r_tic)
        r_tic = time.time()

    logger.info(f'infering {model_name} finished')