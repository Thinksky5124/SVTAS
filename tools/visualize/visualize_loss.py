'''
Author       : Thyssen Wen
Date         : 2023-04-07 15:09:04
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-12 15:38:25
Description  : ref:https://github.com/tomgoldstein/loss-landscape
FilePath     : /SVTAS/tools/visualize/visualize_loss.py
'''
import os
import sys
import torch
import argparse
path = os.path.join(os.getcwd())
sys.path.append(path)
from mmengine.runner import load_state_dict
from svtas.utils.config import Config
import svtas.model.builder as model_builder
import svtas.metric.builder as metric_builder
import svtas.loader.builder as dataset_builder
from svtas.utils.logger import get_logger, setup_logger, coloring
from loss_landspace import (create_random_directions, plot_landspace_1D_loss_err,
                            calulate_loss_landscape, plot_landspace_2D_loss_err, caculate_trajectory,
                            plot_contour_trajectory)
from svtas.engine.extract_engine import LossLandSpaceRunner

def visulizer(cfg, outpath, weight_path,
              xmin=-1, xmax=1, xnum=10, ymin=-1, ymax=1, ynum=10,
              vmin = 0, vmax = 100, vlevel = 0.5,
              plot_1D=False, plot_optimization_path=False):
    outpath = os.path.join(outpath, cfg.model_name)
    isExists = os.path.exists(outpath)
    if not isExists:
        os.makedirs(outpath)
        print(outpath + ' created successful')
    logger = get_logger("SVTAS")

    # construct model
    model = model_builder.build_model(cfg.MODEL).cpu()
    criterion = model_builder.build_loss(cfg.MODEL.loss).cpu()
    rand_directions = create_random_directions(model, plot_1D=plot_1D)

    model.cuda()
    criterion.cuda()
    # prepare trained model
    if weight_path.endswith(".pt"):
        checkpoint = torch.load(weight_path)
    else:
        checkpoint = torch.load(os.path.join(weight_path, cfg.model_name + "_best.pt"))
    state_dicts = checkpoint["model_state_dict"]
    load_state_dict(model, state_dicts, logger=logger)

    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE.test)
    Pipeline = dataset_builder.build_pipline(cfg.PIPELINE.test)
    dataset_config = cfg.DATASET.test
    dataset_config['pipeline'] = Pipeline
    dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    dataset_config['video_batch_size'] = video_batch_size
    test_dataloader = torch.utils.data.DataLoader(
        dataset_builder.build_dataset(dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)
    if plot_1D:
        sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE.train)
        Pipeline = dataset_builder.build_pipline(cfg.PIPELINE.train)
        dataset_config = cfg.DATASET.train
        dataset_config['pipeline'] = Pipeline
        dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
        dataset_config['video_batch_size'] = video_batch_size
        train_dataloader = torch.utils.data.DataLoader(
            dataset_builder.build_dataset(dataset_config),
            batch_size=temporal_clip_batch_size,
            num_workers=test_num_workers,
            collate_fn=sliding_concate_fn)
        train_dataloader.dataset._viodeo_sample_shuffle()
    need_grad_accumulate = cfg.OPTIMIZER.get("need_grad_accumulate", True)
    criterion_metric_name = cfg.get("criterion_metric_name", "F1@0.50")
    
    # add params to metrics
    metric_cfg = cfg.METRIC
    Metric = dict()
    for k, v in metric_cfg.items():
        Metric[k] = metric_builder.build_metric(v)
        
    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = LossLandSpaceRunner(logger=logger, model=model, post_processing=post_processing, criterion=criterion, Metric=Metric,
                                 need_grad_accumulate=need_grad_accumulate, out_path=outpath, logger_interval=cfg.get('logger_interval', 10))
    
    if not plot_1D:
        calulate_loss_landscape(model, rand_directions, outpath, logger, runner, test_dataloader, False, criterion_metric_name, key='test',
                                xmin=xmin, xmax=xmax, xnum=xnum, ymin=ymin, ymax=ymax, ynum=ynum)
        plot_landspace_2D_loss_err(outpath, logger, criterion_metric_name, vmin = vmin, vmax = vmax, vlevel = vlevel, surf_name = "test_loss")
    else:
        calulate_loss_landscape(model, rand_directions, outpath, logger, runner, test_dataloader, True, criterion_metric_name, key='test',
                            xmin=xmin, xmax=xmax, xnum=xnum, ymin=ymin, ymax=ymax, ynum=ynum)
        calulate_loss_landscape(model, rand_directions, outpath, logger, runner, train_dataloader, True, criterion_metric_name, key='train',
                            xmin=xmin, xmax=xmax, xnum=xnum, ymin=ymin, ymax=ymax, ynum=ynum)
        plot_landspace_1D_loss_err(outpath, logger, criterion_metric_name, xmin=xmin, xmax=xmax, loss_max=5, log=False, show=False)
    if plot_optimization_path:
        caculate_trajectory(cfg, weight_path, model, outpath, logger, dir_type='weights', ignore_bias=False)
        plot_contour_trajectory(outpath, logger, surf_name='loss_vals', vmin=vmin, vmax=vmax, vlevel=vlevel, show=False)
    logger.info(coloring("Finish plot loss landspace!"))

def get_args():
    parser = argparse.ArgumentParser("SVTAS visulization optimization script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-w',
                        '--weight',
                        type=str,
                        help='weight file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        default='./output/visulize_loss',
                        help='extract flow file out path')
    parser.add_argument("--plot_1D",
                        action="store_true",
                        help="wheather plot 1D fig")
    parser.add_argument("--plot_optimization_path",
                        action="store_true",
                        help="wheather plot optimization path in 2D loss landspace")

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args

if __name__ == "__main__":
    args = get_args()
    setup_logger(args.out_path, name="SVTAS", level="INFO", tensorboard=False)
    cfg = Config.fromfile(args.config)
    visulizer(cfg, args.out_path, args.weight, 
              xmin=-0.5, xmax=0.5, xnum=51, ymin=-0.5, ymax=0.5, ynum=51,
              vmin = 0, vmax = 10, vlevel = 0.05,
              plot_1D=args.plot_1D, plot_optimization_path=args.plot_optimization_path)
