'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors: Thyssen Wen
LastEditTime: 2022-03-26 14:34:49
Description: test script api
FilePath: /ETETS/tasks/test.py
'''
def test(cfg, weights):
    pass

import os.path as osp
import time
import numpy as np
import torch
from utils.logger import get_logger, AverageMeter, log_batch
from .runner import testRunner

from model.etets import ETETS
from model.loss import ETETSLoss
from dataset.segmentation_dataset import SegmentationDataset
from utils.metric import SegmentationMetric
from dataset.pipline import Pipeline
from dataset.pipline import BatchCompose
from model.post_processing import PostProcessing

@torch.no_grad()
def test(cfg, distributed, weights):
    logger = get_logger("ETETS")
    # 1. Construct model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.construct model
    model = ETETS(**cfg.MODEL).to(device)
    criterion = ETETSLoss(**cfg.MODEL.loss).to(device)

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = BatchCompose(**cfg.COLLATE)
    test_Pipeline = Pipeline(**cfg.PIPELINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_Pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size
    test_dataloader = torch.utils.data.DataLoader(
        SegmentationDataset(**test_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)

    state_dicts = torch.load(weights)['model_state_dict']
    model.load_state_dict(state_dicts)

    # add params to metrics
    Metric = SegmentationMetric(**cfg.METRIC)

    post_processing = PostProcessing(
        num_classes=cfg.MODEL.head.num_classes,
        clip_seg_num=cfg.MODEL.neck.clip_seg_num,
        sliding_window=cfg.DATASET.test.sliding_window,
        sample_rate=cfg.DATASET.test.sample_rate,
        clip_buffer_num=cfg.MODEL.neck.clip_buffer_num)

    runner = testRunner(logger=logger,
                video_batch_size=video_batch_size,
                Metric=Metric,
                cfg=cfg,
                model=model,
                post_processing=post_processing)

    runner.epoch_init()

    for i, data in enumerate(test_dataloader):
        runner.test_one_iter(data=data)

    Metric.accumulate()