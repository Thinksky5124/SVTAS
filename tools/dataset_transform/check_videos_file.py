'''
Author       : Thyssen Wen
Date         : 2022-11-23 19:32:52
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-23 21:43:55
Description  : check video file
FilePath     : /SVTAS/tools/dataset_transform/check_videos_file.py
'''
import argparse
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import torch
from svtas.utils.config import Config
from svtas.utils.logger import get_logger, setup_logger
from svtas.utils.logger import get_logger
from svtas.loader.builder import build_dataset
from svtas.loader.builder import build_pipline

def parse_args():
    parser = argparse.ArgumentParser("SVTAS tools script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(-1)
    return args

def main():
    args = parse_args()
    setup_logger(f"./output/check_videos", name="SVTAS", level="INFO", tensorboard=False)
    cfg = Config.fromfile(args.config)
    logger = get_logger("SVTAS")

    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    num_workers = cfg.DATASET.get('num_workers', 0)

    train_Pipeline = build_pipline(cfg.PIPELINE.train)
    val_Pipeline = build_pipline(cfg.PIPELINE.test)

    # Construct Dataset
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_Pipeline
    train_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    train_dataset_config['video_batch_size'] = video_batch_size
    train_dataset_config['local_rank'] = -1
    train_dataset_config['nprocs'] = 1
    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(train_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=build_pipline(cfg.COLLATE.train))

    val_dataset_config = cfg.DATASET.test
    val_dataset_config['pipeline'] = val_Pipeline
    val_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    val_dataset_config['video_batch_size'] = video_batch_size
    val_dataset_config['local_rank'] = -1
    val_dataset_config['nprocs'] = 1
    val_dataloader = torch.utils.data.DataLoader(
        build_dataset(val_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=build_pipline(cfg.COLLATE.test))
    out_data = dict(vid_list=[])

    logger_interal = 100
    train_dataloader.dataset._viodeo_sample_shuffle()
    cnt = 0
    logger.info("Start check!")
    train_dataloader_iterator = train_dataloader.__iter__()
    while True:
        try:
            data = train_dataloader_iterator.next()
            out_data = data
            cnt += 1
            if cnt % logger_interal == 0:
                step = out_data[0]['step']
                logger.info("Step: " + str(step) + " now check file: " + ",".join(out_data[0]['vid_list']))
                cnt = 0
        except StopIteration:
            logger.info("End train dataset check!")
            break
        except Exception:
            logger.error("Error video file: " + ",".join(out_data[0]['vid_list']))
            continue
    
    val_dataloader_iterator = val_dataloader.__iter__()
    while True:
        try:
            data = val_dataloader_iterator.next()
            out_data = data
            cnt += 1
            if cnt % logger_interal == 0:
                step = out_data[0]['step']
                logger.info("Step: " + str(step) + " now check file: " + ",".join(out_data[0]['vid_list']))
                cnt = 0
        except StopIteration:
            logger.info("End test dataset check!")
            break
        except Exception:
            logger.error("Error video file: " + ",".join(out_data[0]['vid_list']))
            continue

    logger.info("Check finish!")


if __name__ == '__main__':
    main()
