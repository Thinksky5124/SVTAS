def test(cfg, weights):
    pass

import os.path as osp
import time
import numpy as np
import torch
from utils.logger import get_logger, AverageMeter, log_batch
from utils.save_load import mkdir

from model.etets import ETETS
from dataset.segmentation_dataset import SegmentationDataset
from utils.metric import SegmentationMetric
from dataset.pipline import Pipeline

logger = get_logger("ETETS")


def test(cfg, weights):
    with torch.no_grad():
        # 1. Construct model.
        # model = ETETS(cfg.MODEL).cuda()
        # criterion = TotalLoss()

        # 2. Construct dataset and dataloader.
        batch_size = cfg.DATASET.get("test_batch_size", 8)

        # default num worker: 0, which means no subprocess will be created
        num_workers = cfg.DATASET.get('num_workers', 0)

        model.eval()

        state_dicts = torch.load(weights)
        model.set_state_dict(state_dicts)

        # add params to metrics
        cfg.METRIC.data_size = len(dataset)
        cfg.METRIC.batch_size = batch_size
        Metric = SegmentationMetric(cfg.METRIC)
        test_pipeline = Pipeline(cfg.PIPELINE.train)
        
        for idx in range(0):
            test_dataloader = torch.utils.data.DataLoader(
                SegmentationDataset(

                ),
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn_cfg=cfg.get('MIX', None),
                shuffle=False
            )
            # videos sliding stream train
            for i, data in enumerate(test_dataloader):
                # test segment
                outputs = model(data)

                post_precessing()
                
            f1 = Metric.update(i, data, outputs)
        Metric.accumulate()