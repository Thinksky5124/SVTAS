'''
Author: Thyssen Wen
Date: 2022-03-18 19:25:14
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 11:23:50
Description: main script
FilePath     : /SVTAS/tools/launch.py
'''
import argparse
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import random

import numpy as np
import torch

from svtas.tasks.extract import extract
from svtas.tasks.infer import infer
from svtas.tasks.test import test
from svtas.tasks.train import train
from svtas.tasks.profile import profile
from svtas.utils.config import get_config
from svtas.utils.logger import get_logger


def parse_args():
    parser = argparse.ArgumentParser("SVTAS train script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('--mode',
                        '-m',
                        choices=["train", "test", "infer", "profile", 'visulaize', 'extract'],
                        help='run mode')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('--local_rank',
        default=-1,
        type=int,
        help='node rank for distributed training')
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='fixed all random seeds when the program is running')
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='whether to use benchmark to reproduct')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        nprocs = 1
    else:
        nprocs = torch.cuda.device_count()
    # set seed if specified
    seed = args.seed
    if seed is not None:
        assert isinstance(
            seed, int), f"seed must be a integer when specified, but got {seed}"
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.deterministic = True
        # weather accelerate conv op
        torch.backends.cudnn.benchmark = args.benchmark
        logger = get_logger("SVTAS")
        logger.info("Current Seed is: " + str(seed))

    if args.mode in ["test"]:
        test(cfg,
             args=args,
             local_rank=args.local_rank,
             nprocs=nprocs)
    elif args.mode in ["train"]:
        train(cfg,
            args=args,
            local_rank=args.local_rank,
            nprocs=nprocs)
    elif args.mode in ["infer"]:
        infer(cfg,
            args=args,
            local_rank=args.local_rank,
            nprocs=nprocs)
    elif args.mode in ["profile"]:
        profile(cfg,
            args=args,
            local_rank=args.local_rank,
            nprocs=nprocs)
    elif args.mode in ["extract"]:
        extract(cfg,
                args=args,
                local_rank=args.local_rank,
                nprocs=nprocs)
    else:
        raise NotImplementedError(args.mode + " mode not support!")


if __name__ == '__main__':
    main()
