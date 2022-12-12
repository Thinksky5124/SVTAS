'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 23:02:48
Description: train script api
FilePath     : /SVTAS/svtas/tasks/train.py
'''
import os.path as osp
import time

import torch
import torch.distributed as dist
from ..utils.logger import get_logger, log_epoch, tenorboard_log_epoch
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

from ..runner.runner import Runner
import warnings
try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP
except:
    warnings.warn("Can't use apex to accelerate")

def train(cfg,
          args,
          local_rank,
          nprocs,
          use_amp=False,
          weights=None,
          validate=True,):
    """Train model entry
    """
    
    logger = get_logger("SVTAS")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("SVTAS", tensorboard=args.use_tensorboard)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    need_grad_accumulate = cfg.OPTIMIZER.get('need_grad_accumulate', True)

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    criterion_metric_name = cfg.get("criterion_metric_name", "F1@0.50")
    mkdir(output_dir)

    # wheather use amp
    if use_amp is True:
        logger.info("use amp")
        amp.register_float_function(torch.nn, 'ReLU6')
        
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.construct model
        model = build_model(cfg.MODEL).to(device)
        criterion = build_loss(cfg.MODEL.loss).to(device)

        # 2. Construct solver.
        optimizer_cfg = cfg.OPTIMIZER
        optimizer_cfg['model'] = model
        optimizer = build_optimizer(optimizer_cfg)
        # grad to zeros
        optimizer.zero_grad()

        # wheather to use amp
        if use_amp is True:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = build_model(cfg.MODEL).cuda(local_rank)
        criterion = build_loss(cfg.MODEL.loss).cuda(local_rank)

        # 2. Construct solver.
        optimizer_cfg = cfg.OPTIMIZER
        optimizer_cfg['model'] = model
        optimizer = build_optimizer(optimizer_cfg)
        # grad to zeros
        optimizer.zero_grad()
    
        # wheather to use amp
        if use_amp is True:
            device = torch.device('cuda:{}'.format(local_rank))
            model = convert_syncbn_model(model).to(device)
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    # 3. build metirc
    metric_cfg = cfg.METRIC
    Metric = dict()
    for k, v in metric_cfg.items():
        v['train_mode'] = True
        Metric[k] = build_metric(v)

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        path_checkpoint = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}" + ".pt")
        
        if local_rank < 0:
            checkpoint = torch.load(path_checkpoint)
        else:
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(path_checkpoint, map_location=map_location)

        if nprocs > 1:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if use_amp is True:
            amp.load_state_dict(checkpoint['amp'])
        resume_epoch = start_epoch
    # 4. construct Pipeline
    train_Pipeline = build_pipline(cfg.PIPELINE.train)
    val_Pipeline = build_pipline(cfg.PIPELINE.test)
    scheduler_cfg = cfg.LRSCHEDULER
    scheduler_cfg['optimizer'] = optimizer
    scheduler = build_lr_scheduler(scheduler_cfg)
    grad_clip_cfg = cfg.get("GRADCLIP", None)
    if grad_clip_cfg is not None:
        grad_clip = build_optimizer(grad_clip_cfg)
    else:
        grad_clip = None

    # wheather batch train
    batch_train = False
    if cfg.COLLATE.train.name in ["BatchCompose"]:
        batch_train = True
    batch_test = False
    if cfg.COLLATE.test.name in ["BatchCompose"]:
        batch_test = True
    # 5. Construct Dataset
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

    # 6. Train Model
    record_dict = build_recod(cfg.MODEL.architecture, mode="train")

    # 7. Construct post precesing
    post_processing = build_post_precessing(cfg.POSTPRECESSING)

    # construct train runner
    runner = Runner(optimizer=optimizer,
                grad_clip=grad_clip,
                logger=logger,
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
                need_grad_accumulate=need_grad_accumulate)
    best = 0.0
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue

        runner.epoch_init()

        # shuffle video data
        train_dataloader.dataset._viodeo_sample_shuffle()
        r_tic = time.time()
        for i, data in enumerate(train_dataloader):
            if batch_train is True:
                runner.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
            elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
                runner.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
            else:
                break
            r_tic = time.time()
            
        if local_rank <= 0:
            # metric output
            for k, v in runner.Metric.items():
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
            tenorboard_log_epoch(record_dict, epoch + 1, "train", writer=tensorboard_writer)

        def evaluate(best):
            record_dict = build_recod(cfg.MODEL.architecture, mode="validation")
            runner = Runner(logger=logger,
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
                runner_mode='validation')

            # model logger init
            runner.epoch_init()
            r_tic = time.time()
            for i, data in enumerate(val_dataloader):
                if batch_test is True:
                    runner.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
                elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
                    runner.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
                else:
                    break
                r_tic = time.time()

            best_flag = False
            if local_rank <= 0:
                # metric output
                Metric_dict = dict()
                for k, v in runner.Metric.items():
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
                tenorboard_log_epoch(record_dict, epoch + 1, "val", writer=tensorboard_writer)
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
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "cfg": cfg.text}
                else:
                    checkpoint = {"model_state_dict": model_weight_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "amp": amp.state_dict(),
                            "epoch": epoch,
                            "cfg": cfg.text}
                torch.save(checkpoint,
                    osp.join(output_dir, model_name + "_best.pt"))
                logger.info(
                        "Already save the best model (" + criterion_metric_name + f"){int(best * 10000) / 10000}"
                    )

        # 6. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1 and local_rank <= 0:
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
        
        if local_rank >= 0:
            torch.distributed.barrier()      

    if local_rank >= 0:
        dist.destroy_process_group()

    logger.info(f'training {model_name} finished')