'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors: Thyssen Wen
LastEditTime: 2022-04-20 16:36:26
Description: train script api
FilePath: /ETESVS/tasks/train.py
'''
import os.path as osp
import time

import torch
import torch.distributed as dist
from utils.logger import get_logger, AverageMeter, log_epoch, tenorboard_log_epoch
from utils.save_load import mkdir
import model.builder as builder

from dataset.segmentation_dataset import SegmentationDataset
from utils.metric import SegmentationMetric
from dataset.pipline import Pipeline
from dataset.pipline import BatchCompose
from model.post_precessings.etesvs_post_processing import PostProcessing
from .runner import Runner

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP
except:
    pass

def train(cfg,
          args,
          local_rank,
          nprocs,
          use_amp=False,
          weights=None,
          validate=True,):
    """Train model entry
    """
    
    logger = get_logger("ETESVS")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("ETESVS", tensorboard=args.use_tensorboard)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    weight_decay = cfg.OPTIMIZER.get('weight_decay', 0.0005)

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output/{model_name}")
    mkdir(output_dir)

    # wheather use amp
    if use_amp is True:
        logger.info("use amp")
        amp.register_float_function(torch.nn, 'ReLU6')
        
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.construct model
        model = builder.build_model(cfg.MODEL).cuda()
        criterion = builder.build_loss(cfg.MODEL.loss).cuda()

        # 2. Construct solver.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.learning_rate,
            betas=(0.9, 0.999), weight_decay=weight_decay)
        # grad to zeros
        optimizer.zero_grad()

        # wheather to use amp
        if use_amp is True:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = builder.build_model(cfg.MODEL).cuda(local_rank)
        criterion = builder.build_loss(cfg.MODEL.loss).cuda(local_rank)

        # 2. Construct solver.
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.learning_rate,
            betas=(0.9, 0.999), weight_decay=weight_decay)
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
    metric_cfg['train_mode'] = True
    Metric = SegmentationMetric(**metric_cfg)

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        path_checkpoint = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}" + ".pkl")
        
        if local_rank < 0:
            checkpoint = torch.load(path_checkpoint)
        else:
            # configure map_location properly
            map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
            checkpoint = torch.load(path_checkpoint, map_location=map_location)

        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        if use_amp is True:
            amp.load_state_dict(checkpoint['amp'])
        resume_epoch = start_epoch
    # 4. construct Pipeline
    train_Pipeline = Pipeline(**cfg.PIPELINE.train)
    val_Pipeline = Pipeline(**cfg.PIPELINE.test)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.OPTIMIZER.step_size, gamma=cfg.OPTIMIZER.gamma)

    # 5. Construct Dataset
    sliding_concate_fn = BatchCompose(**cfg.COLLATE)
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_Pipeline
    train_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    train_dataset_config['video_batch_size'] = video_batch_size * nprocs
    train_dataset_config['local_rank'] = local_rank
    train_dataset_config['nprocs'] = nprocs
    train_dataloader = torch.utils.data.DataLoader(
        SegmentationDataset(**train_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=sliding_concate_fn)
    
    if validate:
        val_dataset_config = cfg.DATASET.test
        val_dataset_config['pipeline'] = val_Pipeline
        val_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
        val_dataset_config['video_batch_size'] = video_batch_size * nprocs
        val_dataset_config['local_rank'] = local_rank
        val_dataset_config['nprocs'] = nprocs
        val_dataloader = torch.utils.data.DataLoader(
            SegmentationDataset(**val_dataset_config),
            batch_size=temporal_clip_batch_size,
            num_workers=num_workers,
            collate_fn=sliding_concate_fn)

    # 6. Train Model
    record_dict = {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'lr': AverageMeter('lr', 'f', need_avg=False),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                   'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                   'neck_loss': AverageMeter("neck_loss", '.5f'),
                   'head_loss': AverageMeter("head_loss", '.5f')
                  }

    # 7. Construct post precesing
    post_processing = PostProcessing(
        num_classes=cfg.MODEL.head.num_classes,
        clip_seg_num=cfg.MODEL.neck.clip_seg_num,
        sliding_window=cfg.DATASET.train.sliding_window,
        sample_rate=cfg.DATASET.train.sample_rate)

    # construct train runner
    runner = Runner(optimizer=optimizer,
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
                local_rank=local_rank)

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
            runner.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
            r_tic = time.time()
            
        if local_rank <= 0:
            # metric output
            runner.Metric.accumulate()
        
        if local_rank >= 0:
            torch.distributed.barrier()

        # update lr
        scheduler.step()
        ips = "avg_ips: {:.5f} instance/sec.".format(
            video_batch_size * record_dict["batch_time"].count /
            record_dict["batch_time"].sum)
        log_epoch(record_dict, epoch + 1, "train", ips, logger)
        if args.use_tensorboard and local_rank <= 0:
            tenorboard_log_epoch(record_dict, epoch + 1, "train", writer=tensorboard_writer)

        def evaluate(best):
            record_dict = {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                   'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                   'neck_loss': AverageMeter("neck_loss", '.5f'),
                   'head_loss': AverageMeter("head_loss", '.5f')
                  }
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
                # videos sliding stream train
                runner.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
                r_tic = time.time()

            best_flag = False
            if local_rank <= 0:
                # metric output
                Metric_dict = runner.Metric.accumulate()
                
                if Metric_dict["F1@0.50"] > best:
                    best = Metric_dict["F1@0.50"]
                    best_flag = True

            ips = "avg_ips: {:.5f} instance/sec.".format(
                video_batch_size * record_dict["batch_time"].count /
                record_dict["batch_time"].sum)
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
                            "epoch": epoch}
                else:
                    checkpoint = {"model_state_dict": model_weight_dict,
                            "optimizer_state_dict": optimizer.state_dict(),
                            "amp": amp.state_dict(),
                            "epoch": epoch}
                torch.save(checkpoint,
                    osp.join(output_dir, model_name + "_best.pkl"))
                logger.info(
                        f"Already save the best model (F1@0.50){int(best * 10000) / 10000}"
                    )
            
            if local_rank >= 0:
                torch.distributed.barrier()

        # 6. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1 and local_rank <= 0:
            if nprocs > 1:
                model_weight_dict = model.module.state_dict()
            else:
                model_weight_dict = model.state_dict()
            if use_amp is False:
                checkpoint = {"model_state_dict": model_weight_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "epoch": epoch}
            else:
                checkpoint = {"model_state_dict": model_weight_dict,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "amp": amp.state_dict(),
                        "epoch": epoch}
            torch.save(
                checkpoint,
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch + 1:05d}.pkl"))
        
        if local_rank >= 0:
            torch.distributed.barrier()      

    if local_rank >= 0:
        dist.destroy_process_group()

    logger.info(f'training {model_name} finished')