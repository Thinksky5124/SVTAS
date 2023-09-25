'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:44:48
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 10:44:17
Description  : file content
FilePath     : /SVTAS/svtas/engine/normal_engine.py
'''
import time
import os.path as osp
from svtas.utils.logger import log_batch, AverageMeter, get_logger
from .base_engine import BaseEngine
from svtas.model_pipline import BaseModelPipline
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine')
class TrainingBaseEngine(BaseEngine):
    def __init__(self,
                 logger,
                 video_batch_size,
                 Metric,
                 cfg,
                 model_pipline_cfg,
                 record_dict=None,
                 grad_clip=None,
                 use_amp=False,
                 nprocs=1,
                 local_rank=-1,
                 need_grad_accumulate=True):
        super().__init__()
        self.model_pipline: BaseModelPipline = build_model_pipline(**model_pipline_cfg)
        self.grad_clip = grad_clip
        self.logger = logger
        self.video_batch_size = video_batch_size
        self.Metric = Metric
        self.record_dict = record_dict
        self.cfg = cfg
        self.nprocs = nprocs
        self.local_rank = local_rank
        self.use_amp = use_amp
        self.need_grad_accumulate = need_grad_accumulate
    
    def _init_loss_dict(self):
        self.loss_dict = {}
        if self.record_dict is not None:
            for key, value in self.record_dict.items():
                if key.endswith("loss"):
                    if not self.need_grad_accumulate:
                        setattr(value, "output_mean", True)
                        self.loss_dict[key] = value
                    else:
                        self.loss_dict[key] = 0.
    
    def _update_loss_dict(self, input_loss_dict):
        for key, value in self.loss_dict.items():
            if not isinstance(value, AverageMeter):
                self.loss_dict[key] = self.loss_dict[key] + input_loss_dict[key].detach().clone()
            else:
                self.loss_dict[key].update(input_loss_dict[key].detach().clone())

    def _log_loss_dict(self):
        for key, value in self.loss_dict.items():
                if not isinstance(value, AverageMeter):
                    self.record_dict[key].update(value.item(), self.video_batch_size)

    def epoch_init(self):
        # batch videos sampler
        self._init_loss_dict()
        self.seg_acc = 0.
        self.post_processing.init_flag = False
        self.current_step = 0
    
    def resume(self, resume_cfg_dict):
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

        if "optimizer_state_dict" in checkpoint.keys():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        start_epoch = checkpoint['epoch']
        if use_amp is True:
            amp.load_state_dict(checkpoint['amp'])
        return start_epoch
    
    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        end_info_dict = dict(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)
        # get pred result
        out_put_dict = self.model_pipline.after_forward('batch', end_info_dict)

        for k, v in self.Metric.items():
            acc = v.update(out_put_dict['vid'], out_put_dict['ground_truth'], out_put_dict['outputs'])
        
        # logger
        if self.runner_mode in ['train']:
            self.record_dict['lr'].update(self.model_pipline.optimizer.state_dict()['param_groups'][0]['lr'], self.video_batch_size)
        
        self._log_loss_dict()
        self.record_dict['batch_time'].update(time.time() - self.b_tic)
        self.record_dict['Acc'].update(acc, self.video_batch_size)
        self.record_dict['Seg_Acc'].update(self.seg_acc, self.video_batch_size)

        self._init_loss_dict()
        self.seg_acc = 0.

        if self.current_step % self.cfg.get("log_interval", 10) == 0:
            ips = "ips: {:.5f} instance/sec.".format(
                self.video_batch_size / self.record_dict["batch_time"].val)
            if self.runner_mode in ['train']:
                log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "train", ips, self.logger)
            elif self.runner_mode in ['validation']:
                log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "validation", ips, self.logger)
            elif self.runner_mode in ['test']:
                log_batch(self.record_dict, self.current_step, 1, 1, "test", ips, self.logger)

        self.b_tic = time.time()

        self.current_step = step
    
    def _model_forward(self, data_dict):
        score, loss_dict = self.model_pipline.forward(data_dict)
        return score, loss_dict

    def run_one_clip(self, data_dict):
        score, loss_dict = self._model_forward(data_dict)
        data_dict['score'] = score
        self.seg_acc += self.model_pipline.after_forward('iter', data_dict)

        # logger loss
        self._update_loss_dict(loss_dict)

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
            self.post_processing.init_flag = False

@ENGINE.register()
class TrainEngine(TrainingBaseEngine):
    @property
    def runner_mode():
        return 'train'
    
    def epoch_init(self):
        super().epoch_init()
        self.model.train()
        self.b_tic = time.time()
        # reset recoder
        for _, recod in self.record_dict.items():
            recod.reset()
    
    def _model_forward(self, data_dict):
        score, loss_dict = self.model_pipline.forward(data_dict)
        self.model_pipline.caculate_loss(loss_dict)
        return score, loss_dict
    
    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        self.model_pipline.update_model()
        super().batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)

@ENGINE.register()
class ValidationEngine(TrainingBaseEngine):
    @property
    def runner_mode():
        return 'validation'

    def epoch_init(self):
        super().epoch_init()
        self.model.eval()
        self.b_tic = time.time()
        # reset recoder
        for _, recod in self.record_dict.items():
            recod.reset()

@ENGINE.register()
class TestEngine(TrainingBaseEngine):
    @property
    def runner_mode():
        return 'test'

    def epoch_init(self):
        super().epoch_init()
        self.model.eval()
        self.b_tic = time.time()
        # reset recoder
        for _, recod in self.record_dict.items():
            recod.reset()