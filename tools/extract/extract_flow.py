'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:37:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-23 11:33:09
Description  : Extract flow script
FilePath     : /SVTAS/tools/extract/extract_flow.py
'''

import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import torch
import svtas.model.builder as model_builder
import argparse
from svtas.utils.logger import get_logger, setup_logger
from svtas.utils.config import Config
import svtas.loader.builder as dataset_builder
from svtas.engine.extract_engine_raw import ExtractOpticalFlowRunner

@torch.no_grad()
def extractor(cfg, args):
    logger = get_logger("SVTAS")
    
    # construct model
    model = model_builder.build_model(cfg.MODEL).cuda()

    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 1)

    assert video_batch_size == 1, "Only support 1 batch size"

    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE.test)
    Pipeline = dataset_builder.build_pipline(cfg.DATASETPIPLINE)
    dataset_config = cfg.DATASET.config
    dataset_config['pipeline'] = Pipeline
    dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    dataset_config['video_batch_size'] = video_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset_builder.build_dataset(dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)
    
    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = ExtractOpticalFlowRunner(logger=logger, model=model, post_processing=post_processing, out_path=args.out_path, logger_interval=cfg.get('logger_interval', 10))

    runner.epoch_init()
    for i, data in enumerate(dataloader):
        runner.run_one_iter(data=data)
    
    logger.info("Finish all extracting!")

def parse_args():
    parser = argparse.ArgumentParser("SVTAS extract optical flow script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_logger(f"./output/etract_flow", name="SVTAS", level="INFO", tensorboard=False)
    extractor(cfg, args)

if __name__ == '__main__':
    main()