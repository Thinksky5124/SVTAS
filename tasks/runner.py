'''
Author: Thyssen Wen
Date: 2022-03-21 15:22:51
LastEditors: Thyssen Wen
LastEditTime: 2022-03-26 19:48:29
Description: runner script
FilePath: /ETESVS/tasks/runner.py
'''
import torch
import time
from utils.logger import log_batch
import torch.distributed as dist

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # sum-up as the all-reduce operation
    rt /= nprocs # NOTE this is necessary, since all_reduce here do not perform average 
    return rt

class TrainRunner(object):
    def __init__(self,
                 optimizer,
                 logger,
                 video_batch_size,
                 Metric,
                 record_dict,
                 cfg,
                 model,
                 criterion,
                 post_processing,
                 nprocs=1):
        self.optimizer = optimizer
        self.logger = logger
        self.video_batch_size = video_batch_size
        self.Metric = Metric
        self.record_dict = record_dict
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.post_processing = post_processing
        self.nprocs = nprocs
    
    def epoch_init(self):
        # batch videos sampler
        self.videos_loss = 0.
        self.video_seg_loss = 0.
        self.video_cls_loss = 0.
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.model.train()
        self.b_tic = time.time()

    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        self.optimizer.step()
        self.optimizer.zero_grad()

        # clear memery buffer
        # self.model.neck.memery._clear_memery_buffer()

        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
        f1 = self.Metric.update(self.current_step_vid_list, ground_truth_list, outputs)

        self.current_step_vid_list = vid_list
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

        # logger
        self.record_dict['batch_time'].update(time.time() - self.b_tic)
        self.record_dict['loss'].update(self.video_seg_loss + self.video_cls_loss, self.video_batch_size)
        self.record_dict['lr'].update(self.optimizer.state_dict()['param_groups'][0]['lr'], self.video_batch_size)
        self.record_dict['F1@0.5'].update(f1)
        self.record_dict['cls_loss'].update(self.video_cls_loss, self.video_batch_size)
        self.record_dict['seg_loss'].update(self.video_seg_loss, self.video_batch_size)

        self.videos_loss = 0.
        self.video_seg_loss = 0.
        self.video_cls_loss = 0.

        if self.current_step % self.cfg.get("log_interval", 10) == 0:
            ips = "ips: {:.5f} instance/sec.".format(
                self.video_batch_size / self.record_dict["batch_time"].val)
            log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "train", ips, self.logger)
        self.current_step = step
        self.b_tic = time.time()

    def train_one_step(self, imgs, labels, masks, vid_list, sliding_num, idx):
        # move data
        imgs = imgs.cuda()
        masks = masks.cuda()
        labels = labels.cuda()
        # train segment
        outputs = self.model(imgs, masks, idx)
        seg_score, cls_score = outputs
        cls_loss, seg_loss = self.criterion(seg_score, cls_score, masks, labels)
        
        loss = (cls_loss + seg_loss) / sliding_num

        if self.nprocs > 1:
            loss = reduce_mean(loss, self.nprocs)
            cls_loss = reduce_mean(cls_loss, self.nprocs)
            seg_loss = reduce_mean(seg_loss, self.nprocs)

        loss.backward()

        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
            self.post_processing.update(seg_score, labels, idx)
            self.videos_loss = self.videos_loss + loss.item()
            self.video_seg_loss = self.video_seg_loss + seg_loss.item()
            self.video_cls_loss = self.video_cls_loss + cls_loss.item()

    def train_one_iter(self, data, r_tic, epoch):
        # videos sliding stream train
        self.record_dict['reader_time'].update(time.time() - r_tic)
        for sliding_seg in data:
            imgs, labels, masks, vid_list, sliding_num, step, idx = sliding_seg
            # wheather next step
            if self.current_step != step:
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)

            if idx >= 0:
                self.train_one_step(imgs, labels, masks, vid_list, sliding_num, idx)

class valRunner(object):
    def __init__(self,
                 logger,
                 video_batch_size,
                 Metric,
                 record_dict,
                 cfg,
                 model,
                 criterion,
                 post_processing,
                 nprocs=1):
        self.logger = logger
        self.video_batch_size = video_batch_size
        self.Metric = Metric
        self.record_dict = record_dict
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.post_processing = post_processing
        self.nprocs = nprocs
    
    def epoch_init(self):
        # batch videos sampler
        self.videos_loss = 0.
        self.video_seg_loss = 0.
        self.video_cls_loss = 0.
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.model.eval()
        self.b_tic = time.time()

    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        # clear memery buffer
        # self.model.neck.memery._clear_memery_buffer()

        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
        f1 = self.Metric.update(self.current_step_vid_list, ground_truth_list, outputs)

        self.current_step_vid_list = vid_list
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

        # logger
        self.record_dict['batch_time'].update(time.time() - self.b_tic)
        self.record_dict['loss'].update(self.video_seg_loss + self.video_cls_loss, self.video_batch_size)
        self.record_dict['F1@0.5'].update(f1)
        self.record_dict['cls_loss'].update(self.video_cls_loss, self.video_batch_size)
        self.record_dict['seg_loss'].update(self.video_seg_loss, self.video_batch_size)

        self.videos_loss = 0.
        self.video_seg_loss = 0.
        self.video_cls_loss = 0.

        if self.current_step % self.cfg.get("log_interval", 10) == 0:
            ips = "ips: {:.5f} instance/sec.".format(
                self.video_batch_size / self.record_dict["batch_time"].val)
            log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "val", ips, self.logger)
        self.current_step = step
        self.b_tic = time.time()

    def val_one_step(self, imgs, labels, masks, vid_list, sliding_num, idx):
        # move data
        imgs = imgs.cuda()
        masks = masks.cuda()
        labels = labels.cuda()
        # train segment
        outputs = self.model(imgs, masks, idx)
        seg_score, cls_score = outputs
        cls_loss, seg_loss = self.criterion(seg_score, cls_score, masks, labels)
        
        loss = (cls_loss + seg_loss) / sliding_num

        if self.nprocs > 1:
            loss = reduce_mean(loss, self.nprocs)
            cls_loss = reduce_mean(cls_loss, self.nprocs)
            seg_loss = reduce_mean(seg_loss, self.nprocs)

        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
            self.post_processing.update(seg_score, labels, idx)
            self.videos_loss = self.videos_loss + loss.item()
            self.video_seg_loss = self.video_seg_loss + seg_loss.item()
            self.video_cls_loss = self.video_cls_loss + cls_loss.item()

    def val_one_iter(self, data, r_tic, epoch):
        # videos sliding stream val
        self.record_dict['reader_time'].update(time.time() - r_tic)
        for sliding_seg in data:
            imgs, labels, masks, vid_list, sliding_num, step, idx = sliding_seg
            # wheather next step
            if self.current_step != step:
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)

            if idx >= 0:
                self.val_one_step(imgs, labels, masks, vid_list, sliding_num, idx)

class testRunner(object):
    def __init__(self,
                 logger,
                 video_batch_size,
                 Metric,
                 cfg,
                 model,
                 post_processing,
                 nprocs=1):
        self.logger = logger
        self.video_batch_size = video_batch_size
        self.Metric = Metric
        self.cfg = cfg
        self.model = model
        self.post_processing = post_processing
        self.nprocs = nprocs
    
    def epoch_init(self):
        # batch videos sampler
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.model.eval()

    def batch_end_step(self, sliding_num, vid_list, step):
        # clear memery buffer
        # self.model.neck.memery._clear_memery_buffer()

        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
        f1 = self.Metric.update(self.current_step_vid_list, ground_truth_list, outputs)

        self.current_step_vid_list = vid_list
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

        self.current_step = step

    def test_one_step(self, imgs, labels, masks, vid_list, sliding_num, idx):
        # move data
        imgs = imgs.cuda()
        masks = masks.cuda()
        labels = labels.cuda()
        # train segment
        outputs = self.model(imgs, masks, idx)
        seg_score, cls_score = outputs

        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
            self.post_processing.update(seg_score, labels, idx)

    def test_one_iter(self, data):
        # videos sliding stream val
        for sliding_seg in data:
            imgs, labels, masks, vid_list, sliding_num, step, idx = sliding_seg
            # wheather next step
            if self.current_step != step:
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step)

            if idx >= 0:
                self.test_one_step(imgs, labels, masks, vid_list, sliding_num, idx)