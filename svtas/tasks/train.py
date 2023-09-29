'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 16:12:36
Description: train script api
FilePath     : /SVTAS/svtas/tasks/train.py
'''
import os.path as osp
import time

import torch
import torch.distributed as dist
from ..utils.logger import get_logger, log_epoch, coloring
from ..utils.save_load import mkdir
from ..utils.recorder import build_recod
from ..model.builder import build_model
from ..model.builder import build_loss
from ..loader.builder import build_dataset
from ..loader.builder import build_pipline
from ..metric.builder import build_metric
from ..model.builder import build_post_precessing
from ..optimizer.builder import build_optimizer
from ..optimizer.builder import build_lr_scheduler

from ..engine.builder import build_engine

def train(cfg,
          args,
          local_rank,
          nprocs,
          use_amp=False,
          weights=None,
          validate=True,):
    """Train model entry
    """
    
    # 1. init logger and output folder
    logger = get_logger("SVTAS")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("SVTAS", tensorboard=args.use_tensorboard)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    criterion_metric_name = cfg.get("criterion_metric_name", "F1@0.50")
    best_epoch = 0
    mkdir(output_dir)
    
    # 2. build metirc
    metric_cfg = cfg.METRIC
    Metric = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = True
        Metric[k] = build_metric(v)
    
    # 3. build Recod
    record_dict = build_recod(cfg.MODEL.architecture, mode="train")

    # 4. construct Pipeline
    train_Pipeline = build_pipline(cfg.PIPELINE.train)
    val_Pipeline = build_pipline(cfg.PIPELINE.test)

    # wheather batch train
    batch_train = False
    if cfg.COLLATE.train.name in ["BatchCompose"]:
        batch_train = True
    batch_test = False
    if cfg.COLLATE.test.name in ["BatchCompose"]:
        batch_test = True
        
    # 5. Construct Dataset
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    num_workers = cfg.DATASET.get('num_workers', 0)
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_Pipeline
    train_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    train_dataset_config['video_batch_size'] = video_batch_size * nprocs
    train_dataset_config['local_rank'] = local_rank
    train_dataset_config['nprocs'] = nprocs
    train_dataloader = torch.utils.data.DataLoader(
        build_dataset(train_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=build_pipline(cfg.COLLATE.train))
    
    if validate:
        val_dataset_config = cfg.DATASET.test
        val_dataset_config['pipeline'] = val_Pipeline
        val_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
        val_dataset_config['video_batch_size'] = video_batch_size * nprocs
        val_dataset_config['local_rank'] = local_rank
        val_dataset_config['nprocs'] = nprocs
        val_dataloader = torch.utils.data.DataLoader(
            build_dataset(val_dataset_config),
            batch_size=temporal_clip_batch_size,
            num_workers=num_workers,
            collate_fn=build_pipline(cfg.COLLATE.test))

    # 6. Train
    best = 0.0
    epoch_start_time = time.time()

    # 7. build engine
    engine_config = cfg.ENGINE
    engine_config['logger'] = logger
    engine_config['Metric'] = Metric
    engine_config['Dataloader'] = train_dataloader
    engine = build_engine(engine_config)

    # 8. resume engine
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        resume_cfg_dict = dict()
        engine.resume(resume_cfg_dict)

    # 9. train
    engine.run()
    
    for epoch in range(0, cfg.epochs):
        if epoch <= resume_epoch and resume_epoch != 0:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{resume_epoch+1}], continue... "
            )
            continue

        engine.epoch_init()

        # shuffle video data
        train_dataloader.dataset._viodeo_sample_shuffle()
        r_tic = time.time()
        for i, data in enumerate(train_dataloader):
            if batch_train is True:
                engine.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
            elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
                engine.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
            else:
                break
            r_tic = time.time()
            if args.use_tensorboard and local_rank <= 0:
                tensorboard_writer.update_step()
            
        if local_rank <= 0:
            # metric output
            for k, v in engine.Metric.items():
                v.accumulate()
        
        if local_rank >= 0:
            torch.distributed.barrier()

        # update lr
        scheduler.step()
        ips = "avg_ips: {:.5f} instance/sec.".format(
            video_batch_size * record_dict["batch_time"].count /
            (record_dict["batch_time"].sum + 1e-10))
        log_epoch(record_dict, epoch + 1, "train", ips, logger)
        if args.use_tensorboard and local_rank <= 0:
            tensorboard_writer.tenorboard_log_epoch(record_dict, epoch + 1, "train")

        def evaluate(best):
            record_dict = build_recod(cfg.MODEL.architecture, mode="validation")
            engine = Runner(logger=logger,
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
                engine_mode='validation',
                need_grad_accumulate=need_grad_accumulate)

            # model logger init
            engine.epoch_init()
            r_tic = time.time()
            for i, data in enumerate(val_dataloader):
                if batch_test is True:
                    engine.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
                elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
                    engine.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
                else:
                    break
                r_tic = time.time()

            best_flag = False
            if local_rank <= 0:
                # metric output
                Metric_dict = dict()
                for k, v in engine.Metric.items():
                    temp_Metric_dict = v.accumulate()
                    Metric_dict.update(temp_Metric_dict)
                
                if Metric_dict[criterion_metric_name] > best:
                    best = Metric_dict[criterion_metric_name]
                    best_flag = True

            ips = "avg_ips: {:.5f} instance/sec.".format(
                video_batch_size * record_dict["batch_time"].count /
                (record_dict["batch_time"].sum + 1e-10))
            log_epoch(record_dict, epoch + 1, "val", ips, logger)
            if args.use_tensorboard and local_rank <= 0:
                record_dict.update(Metric_dict)
                tensorboard_writer.tenorboard_log_epoch(record_dict, epoch + 1, "val")
            return best, best_flag

        # 5. Validation
        if validate and (epoch % cfg.get("val_interval", 1) == 0
                         or epoch == cfg.epochs - 1):
            with torch.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                if nprocs > 1:
                    model_weight_dict = model.module.state_dict()
                else:
                    model_weight_dict = model.state_dict()
                if use_amp is False:
                    checkpoint = {"model_state_dict": model_weight_dict,
                            "epoch": epoch,
                            "cfg": cfg.text}
                else:
                    checkpoint = {"model_state_dict": model_weight_dict,
                            "epoch": epoch,
                            "cfg": cfg.text}
                torch.save(checkpoint,
                    osp.join(output_dir, model_name + "_best.pt"))
                logger.info(
                        coloring("Already save the best model (" + criterion_metric_name + f"){int(best * 10000) / 10000}.", "OKGREEN")
                    )
                best_epoch = epoch

        # 6. Save model and optimizer
        if (epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1) and local_rank <= 0:
            if nprocs > 1:
                model_weight_dict = model.module.state_dict()
            else:
                model_weight_dict = model.state_dict()
            if use_amp is False:
                checkpoint = {"model_state_dict": model_weight_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.text}
            else:
                checkpoint = {"model_state_dict": model_weight_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "amp": amp.state_dict(),
                        "epoch": epoch,
                        "cfg": cfg.text}
            torch.save(
                checkpoint,
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch + 1:05d}.pt"))
            logger.info(
                        f"Already save the checkpoint model from epoch: {epoch + 1}."
                    )
        
        # 7. Computer ETA
        epoch_duration_time = time.time() - epoch_start_time
        timeArray = time.gmtime(epoch_duration_time * (cfg.epochs - (epoch + 1)))
        formatTime = f"{timeArray.tm_mday - 1} day : {timeArray.tm_hour} h : {timeArray.tm_min} m : {timeArray.tm_sec} s."
        logger.info(coloring(f"ETA: {formatTime}", 'OKBLUE'))
        epoch_start_time = time.time()

        if local_rank >= 0:
            torch.distributed.barrier()      

    if local_rank >= 0:
        dist.destroy_process_group()

    logger.info(f'training {model_name} finished')
    if validate:
        logger.info(f"The best performance on {criterion_metric_name} is {int(best * 10000) / 10000}, in epoch {best_epoch + 1}.")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer.close()