'''
Author: Thyssen Wen
Date: 2022-03-18 19:25:14
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 22:50:51
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

from svtas.tasks.visualize_loss import visulize_loss
from svtas.tasks.visualize import visualize
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
                        choices=["train", "test", "infer", "profile", 'visualize', 'extract', 'visualize_loss'],
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
    parser.add_argument('--world_size',
        default=1,
        type=int,
        help='num of world size')
    parser.add_argument('--master_addr',
        default="127.0.0.1",
        type=str,
        help='master address')
    parser.add_argument('--master_port',
        default="29500",
        type=str,
        help='master port')
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
        choices=['pytorch', 'torchrun'],
        default='pytorch',
        help='job launcher')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
    else:
        os.environ["MASTER_ADDR"] = args.master_addr
        os.environ["MASTER_PORT"] = args.master_port
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override)

    # init distributed env first, since logger depends on the dist info.
    nprocs = args.world_size
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

    task_func_dict = {
        "train": train,
        "test": test,
        "infer": infer,
        "profile": profile,
        "extract": extract,
        "visualize": visualize,
        "visulize_loss": visulize_loss
    }
    
    if nprocs <= 1:
        # single process run task
        task_func_dict[args.mode](local_rank=args.local_rank, nprocs=nprocs, cfg=cfg, args=args)
    else:
        # multi process run task
        if args.launcher == "pytorch":
            import torch.multiprocessing as mp
            mp.spawn(task_func_dict[args.mode], nprocs=nprocs, args=(nprocs, cfg, args))
        elif args.launcher == "torchrun":
            task_func_dict[args.mode](local_rank=int(os.environ['LOCAL_RANK']), nprocs=int(os.environ['WORLD_SIZE']), cfg=cfg, args=args)
if __name__ == '__main__':
    main()
