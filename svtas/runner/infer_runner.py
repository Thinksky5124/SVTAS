'''
Author       : Thyssen Wen
Date         : 2022-09-24 14:59:32
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 19:13:10
Description  : Infer Runner Class
FilePath     : /SVTAS/svtas/runner/infer_runner.py
'''
import torch
import time
from ..utils.logger import log_batch

from ..utils.logger import get_logger
import numpy as np
from .runner import Runner

class InferONNXRunner(Runner):
    def __init__(self,
                 logger,
                 video_batch_size,
                 Metric,
                 cfg,
                 model,
                 post_processing,
                 record_dict=None,
                 nprocs=1,
                 local_rank=-1,):
        super().__init__(logger,
                         video_batch_size,
                         Metric,
                         cfg,
                         model,
                         post_processing,
                         record_dict=record_dict,
                         criterion=None,
                         optimizer=None,
                         use_amp=False,
                         nprocs=nprocs,
                         local_rank=local_rank)
        self.runner_mode = "infer"
    
    def epoch_init(self):
        # batch videos sampler
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None

        self.b_tic = time.time()
        # reset recoder
        for _, recod in self.record_dict.items():
            recod.reset()
    
    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        # Todos: clear model memory cache
        # # clear memory buffer
        # if self.nprocs > 1:
        #     self.model.module._clear_memory_buffer()
        # else:
        #     self.model._clear_memory_buffer()

        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
        vid = self.current_step_vid_list

        # Todos: distribution infer        
        # if self.nprocs > 1:
        #     collect_dict = dict(
        #         predict=pred_cls_list,
        #         output_np=pred_score_list,
        #         ground_truth=ground_truth_list,
        #         vid=self.current_step_vid_list
        #     )
        #     gather_objects = [collect_dict for _ in range(self.nprocs)] # any picklable object
        #     output_list = [None for _ in range(self.nprocs)]
        #     dist.all_gather_object(output_list, gather_objects[dist.get_rank()])
        #     # collect
        #     pred_cls_list_i = []
        #     pred_score_list_i = []
        #     ground_truth_list_i = []
        #     vid_i = []
        #     for output_dict in output_list:
        #         pred_cls_list_i = pred_cls_list_i + output_dict["predict"]
        #         pred_score_list_i = pred_score_list_i + output_dict["output_np"]
        #         ground_truth_list_i = ground_truth_list_i + output_dict["ground_truth"]
        #         vid_i = vid_i + output_dict["vid"]
        #     outputs = dict(predict=pred_cls_list_i,
        #                     output_np=pred_score_list_i)
        #     ground_truth_list = ground_truth_list_i
        #     vid = vid_i
        
        self.Metric.update(vid, ground_truth_list, outputs)

        self.current_step_vid_list = vid_list
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))
        
        # Todos: distribution infer
        # if self.nprocs > 1:
        #     torch.distributed.barrier()
        #     self._distribute_sync_loss_dict()

        # logger
        self.record_dict['batch_time'].update(time.time() - self.b_tic)

        if self.current_step % self.cfg.get("log_interval", 10) == 0:
            ips = "ips: {:.5f} instance/sec.".format(
                self.video_batch_size / self.record_dict["batch_time"].val)
            log_batch(self.record_dict, self.current_step, 1, 1, "infer", ips, self.logger)

        self.b_tic = time.time()

        self.current_step = step

    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value) and key in self.cfg.INFER.input_names:
                input_data[key] = value.numpy()

        outputs = self.model.run(None, input_data)
        
        score = outputs['output']
            
        return score
        
    def run_one_clip(self, data_dict):
        vid_list = data_dict['vid_list']
        sliding_num = data_dict['sliding_num']
        idx = data_dict['current_sliding_cnt']
        # train segment
        if self.nprocs > 1 and idx < sliding_num - 1:
            # Todos: distribution infer
            # with self.model.no_sync():
            #     # multi-gpus
            #     score = self._model_forward(data_dict)
            raise NotImplementedError
        else:
            # single gpu
            score = self._model_forward(data_dict)
            
        if self.post_processing.init_flag is not True:
            self.post_processing.init_scores(sliding_num, len(vid_list))
            self.current_step_vid_list = vid_list
        self.post_processing.update(score, np.zeros_like(score[0, 0, 0:1, :]).astype(np.int64), idx) / sliding_num
        
    def run_one_iter(self, data, r_tic=None, epoch=None):
        # videos sliding stream train
        self.record_dict['reader_time'].update(time.time() - r_tic)

        for sliding_seg in data:
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step
            if self.current_step != step or (len(vid_list) <= 0 and step == 1):
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)

            if idx >= 0: 
                self.run_one_clip(sliding_seg)
    
    def run_one_batch(self, data, r_tic=None, epoch=None):
        # videos batch train
        self.record_dict['reader_time'].update(time.time() - r_tic)

        for sliding_seg in data:
            step = self.current_step
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']

            # run one batch
            self.run_one_clip(sliding_seg)
            self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)
            self.current_step = self.current_step + 1