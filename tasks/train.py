import os.path as osp
import time

import numpy as np
import torch
from utils.logger import get_logger, AverageMeter, log_batch, log_epoch
from utils.save_load import mkdir

from model.etets import ETETS
from model.loss import ETETSLoss
from dataset.segmentation_dataset import SegmentationDataset
from utils.metric import SegmentationMetric
from dataset.pipline import Pipeline
from dataset.pipline import BatchCompose
from model.post_processing import PostProcessing

def train(cfg,
          distributed,
          weights=None,
          validate=True):
    """Train model entry
    """

    logger = get_logger("ETETS")
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)

    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    model_name = cfg.model_name
    output_dir = cfg.get("output_dir", f"./output")
    mkdir(output_dir)

    if distributed == False:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        local_rank = torch.distributed.get_rank()
        device = torch.device(f'cuda:{local_rank}')
    # 1.construct model
    model = ETETS(**cfg.MODEL).cuda()
    criterion = ETETSLoss(**cfg.MODEL.loss)

    # 2. build metirc
    Metric = SegmentationMetric(**cfg.METRIC)

    # 3. Construct solver.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.OPTIMIZER.learning_rate,
        betas=(0.9, 0.999), weight_decay=0.0005)

    # Resume
    resume_epoch = cfg.get("resume_epoch", 0)
    if resume_epoch:
        path_checkpoint = osp.join(output_dir,
                            model_name + f"_epoch_{resume_epoch:05d}" + ".pkl")
        checkpoint = torch.load(path_checkpoint)

        model.load_state_dict(checkpoint['net'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        resume_epoch = start_epoch
    # 4. construct Pipeline
    train_Pipeline = Pipeline(**cfg.PIPELINE.train)
    val_Pipeline = Pipeline(**cfg.PIPELINE.test)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.OPTIMIZER.step_size, gamma=cfg.OPTIMIZER.gamma)

    # 5. Construct Dataset
    sliding_concate_fn = BatchCompose(**cfg.COLLATE)
    train_dataset_config = cfg.DATASET.train
    train_dataset_config['pipeline'] = train_Pipeline
    train_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    train_dataset_config['video_batch_size'] = video_batch_size
    train_dataloader = torch.utils.data.DataLoader(
        SegmentationDataset(**train_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=num_workers,
        collate_fn=sliding_concate_fn)
    
    if validate:
        val_dataset_config = cfg.DATASET.test
        val_dataset_config['pipeline'] = val_Pipeline
        val_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
        val_dataset_config['video_batch_size'] = video_batch_size
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
                   'cls_loss': AverageMeter("cls_loss", '.5f'),
                   'seg_loss': AverageMeter("seg_loss", '.5f')
                  }

    # 7. Construct post precesing
    post_processing = PostProcessing(
        num_classes=cfg.MODEL.head.num_classes,
        clip_seg_num=cfg.MODEL.neck.clip_seg_num,
        sliding_window=cfg.DATASET.train.sliding_window,
        sample_rate=cfg.DATASET.train.sample_rate,
        clip_buffer_num=cfg.MODEL.neck.clip_buffer_num)

    best = 0.0
    for epoch in range(0, cfg.epochs):
        if epoch < resume_epoch:
            logger.info(
                f"| epoch: [{epoch+1}] <= resume_epoch: [{ resume_epoch}], continue... "
            )
            continue

        model.train()

        # batch videos sampler
        tic = time.time()
        videos_loss = 0.
        video_seg_loss = 0.
        video_cls_loss = 0.
        post_processing.init_flag = False
        current_step = 0
        current_step_vid_list = None
        # shuffle video data
        train_dataloader.dataset._viodeo_sample_shuffle()
        for i, data in enumerate(train_dataloader):
            # videos sliding stream train
            record_dict['reader_time'].update(time.time() - tic)
            for sliding_seg in data:
                imgs, labels, masks, vid_list, sliding_num, step, idx = sliding_seg
                # wheather next step
                if current_step != step:
                    optimizer.step()
                    optimizer.zero_grad()

                    # get pred result
                    pred_score_list, pred_cls_list, ground_truth_list = post_processing.output()
                    outputs = dict(predict=pred_cls_list,
                                    output_np=pred_score_list)
                    f1 = Metric.update(current_step_vid_list, ground_truth_list, outputs)

                    current_step_vid_list = vid_list
                    if len(current_step_vid_list) > 0:
                        post_processing.init_scores(sliding_num, len(vid_list))

                    # logger
                    record_dict['batch_time'].update(time.time() - tic)
                    record_dict['loss'].update(video_seg_loss + video_cls_loss, video_batch_size)
                    record_dict['lr'].update(optimizer.state_dict()['param_groups'][0]['lr'], video_batch_size)
                    record_dict['F1@0.5'].update(f1)
                    record_dict['cls_loss'].update(video_cls_loss, video_batch_size)
                    record_dict['seg_loss'].update(video_seg_loss, video_batch_size)

                    videos_loss = 0.
                    video_seg_loss = 0.
                    video_cls_loss = 0.

                    if current_step % cfg.get("log_interval", 10) == 0:
                        ips = "ips: {:.5f} instance/sec.".format(
                            video_batch_size / record_dict["batch_time"].val)
                        log_batch(record_dict, current_step, epoch + 1, cfg.epochs, "train", ips, logger)
                    current_step = step

                if idx >= 0:
                    # move data
                    imgs = imgs.cuda()
                    masks = masks.cuda()
                    labels = labels.cuda()
                    # train segment
                    outputs = model(imgs, masks)
                    seg_score, cls_score = outputs
                    cls_loss, seg_loss = criterion(seg_score, cls_score, masks, labels)
                    
                    loss = (cls_loss + seg_loss) / sliding_num

                    loss.backward()
                    if post_processing.init_flag is not True:
                        post_processing.init_scores(sliding_num, len(vid_list))
                        current_step_vid_list = vid_list
                    post_processing.update(seg_score, labels, idx)
                    videos_loss += loss.item()
                    video_seg_loss += seg_loss.item()
                    video_cls_loss += cls_loss.item()

                    tic = time.time()

        # metric output
        Metric.accumulate()

        # update lr
        scheduler.step()
        ips = "avg_ips: {:.5f} instance/sec.".format(
            video_batch_size * record_dict["batch_time"].count /
            record_dict["batch_time"].sum)
        log_epoch(record_dict, epoch + 1, "train", ips, logger)

        def evaluate(best):
            model.eval()
            record_dict = {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'cls_loss': AverageMeter("cls_loss", '.5f'),
                   'seg_loss': AverageMeter("seg_loss", '.5f')
                  }

            tic = time.time()
            # model logger init
            post_processing.init_flag = False
            videos_loss = 0.
            video_seg_loss = 0.
            video_cls_loss = 0.
            current_step = 0
            current_step_vid_list = None
            for i, data in enumerate(val_dataloader):
                # videos sliding stream train
                record_dict['reader_time'].update(time.time() - tic)
                for sliding_seg in data:
                    imgs, labels, masks, vid_list, sliding_num, step, idx = sliding_seg
                    # wheather next step
                    if current_step != step:
                        # get pred result
                        pred_score_list, pred_cls_list, ground_truth_list = post_processing.output()
                        outputs = dict(predict=pred_cls_list,
                                        output_np=pred_score_list)
                        f1 = Metric.update(current_step_vid_list, ground_truth_list, outputs)

                        current_step_vid_list = vid_list
                        if len(current_step_vid_list) > 0:
                            post_processing.init_scores(sliding_num, len(vid_list))

                        # logger
                        record_dict['batch_time'].update(time.time() - tic)
                        record_dict['loss'].update(video_seg_loss + video_cls_loss, video_batch_size)
                        record_dict['F1@0.5'].update(f1)
                        record_dict['cls_loss'].update(video_cls_loss, video_batch_size)
                        record_dict['seg_loss'].update(video_seg_loss, video_batch_size)

                        videos_loss = 0.
                        video_seg_loss = 0.
                        video_cls_loss = 0.

                        if current_step % cfg.get("log_interval", 10) == 0:
                            ips = "ips: {:.5f} instance/sec.".format(
                                video_batch_size / record_dict["batch_time"].val)
                            log_batch(record_dict, current_step, epoch + 1, cfg.epochs, "val", ips, logger)
                        current_step = step

                    if idx >= 0:
                        # move data
                        imgs = imgs.cuda()
                        masks = masks.cuda()
                        labels = labels.cuda()
                        # train segment
                        outputs = model(imgs, masks)
                        seg_score, cls_score = outputs
                        cls_loss, seg_loss = criterion(seg_score, cls_score, masks, labels)
                        
                        loss = (cls_loss + seg_loss) / sliding_num

                        if post_processing.init_flag is not True:
                            post_processing.init_scores(sliding_num, len(vid_list))
                            current_step_vid_list = vid_list
                        post_processing.update(seg_score, labels, idx)
                        videos_loss += loss.item()
                        video_seg_loss += seg_loss.item()
                        video_cls_loss += cls_loss.item()

                        tic = time.time()

            # metric output
            Metric_dict = Metric.accumulate()

            ips = "avg_ips: {:.5f} instance/sec.".format(
                video_batch_size * record_dict["batch_time"].count /
                record_dict["batch_time"].sum)
            log_epoch(record_dict, epoch + 1, "val", ips, logger)

            best_flag = False
            if Metric_dict["F1@0.50"] > best:
                best = Metric_dict["F1@0.50"]
                best_flag = True
            return best, best_flag

        # 5. Validation
        if validate and (epoch % cfg.get("val_interval", 1) == 0
                         or epoch == cfg.epochs - 1):
            with torch.no_grad():
                best, save_best_flag = evaluate(best)
            # save best
            if save_best_flag:
                torch.save(model.state_dict(),
                     osp.join(output_dir, model_name + "_best.pkl"))
                logger.info(
                        f"Already save the best model (F1@0.50){int(best * 10000) / 10000}"
                    )

        # 6. Save model and optimizer
        if epoch % cfg.get("save_interval", 1) == 0 or epoch == cfg.epochs - 1:
            torch.save(
                model.state_dict(),
                osp.join(output_dir,
                         model_name + f"_epoch_{epoch + 1:05d}.pkl"))

    logger.info(f'training {model_name} finished')
