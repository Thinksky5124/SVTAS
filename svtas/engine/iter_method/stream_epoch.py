'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:41:13
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 16:51:04
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/stream_epoch.py
'''
from typing import Any
from .epoch import EpochMethod
import time
from svtas.utils import AbstractBuildFactory
from svtas.utils.logger import coloring

@AbstractBuildFactory.register('engine_component')
class StreamEpochMethod(EpochMethod):
    def __init__(self,
                 epoch_num: int,
                 batch_size: int,
                 criterion_metric_name: str,
                 mode: str = 'train',
                 logger_iter_interval: int = 5,
                 logger_epoch_interval: int = 1,
                 save_interval: int = 10,
                 test_interval: int = -1) -> None:
        super().__init__(epoch_num, batch_size, criterion_metric_name, mode,
                         logger_iter_interval, logger_epoch_interval, save_interval,
                         test_interval)
    
    def init_epoch(self, epoch):
        super().init_epoch(epoch)
        self.b_tic = time.time()
        self.current_step = 0
    
    def batch_end_step(self, input_data, epoch):
        # update param
        if self.model_pipline.grad_accumulate is not None:
            self.model_pipline.grad_accumulate.set_update_conf()
        self.model_pipline.update_model_param()
        # post processing
        output_dict = self.model_pipline.output_post_processing()
        self.model_pipline.init_post_processing(input_data=input_data)

        # update metric
        for k, v in self.metric.items():
            acc = v.update(output_dict['vid'], output_dict['ground_truth'], output_dict['outputs'])
        self.record['Acc'].update(acc, len(input_data['vid_list']))

        step = input_data['step']
        self.record['batch_time'].update(time.time() - self.b_tic)
        if self.mode == 'train':
            self.record['lr'].update(self.model_pipline.optimizer.state_dict()['param_groups'][0]['lr'], self.batch_size)
        if step % self.logger_iter_interval == 0:
            self.logger_iter(step, epoch)
        self.b_tic = time.time()
        self.current_step = step

    def run_one_iter(self, data, r_tic=None, epoch=None):
        # videos sliding stream train
        for sliding_seg in data:
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step
            if self.current_step != step or (len(vid_list) <= 0 and step == 1):
                self.batch_end_step(input_data=sliding_seg, epoch=epoch)

            if idx >= 0: 
                outputs, loss_dict = self.run_one_forward(sliding_seg)
                if not self.model_pipline.post_processing.init_flag:
                    self.model_pipline.init_post_processing(input_data=sliding_seg)
                self.model_pipline.update_post_processing(model_outputs=outputs, input_data=sliding_seg)
                self.record.stream_update_dict(loss_dict)

    def end_iter(self, b_tic, step, epoch):
        pass
    
    def init_iter(self, r_tic):
        return super().init_iter(r_tic)
    
    def run(self, *args: Any, **kwds: Any) -> Any:
        """
        run function processing
        ```
        +----------------------+
        |   init run           |
        +----------------------+
        |   epoch pre hook     |
        +----------------------+
        |   start epoch enmu   |
        +----------------------+
        |   init epoch         |
        +----------------------+
        |   iter pre hook      |
        +----------------------+
        |   start iter enmu    |
        +----------------------+
        |   init iter          |
        +----------------------+
        |   run one bactch     |
        +----------------------+
        |   end iter           |
        +----------------------+
        |   end iter enmu      |
        +----------------------+
        |   iter end hook      |
        +----------------------+
        |   end epoch          |
        +----------------------+
        |   epoch end hook     |
        +----------------------+
        |   end run            |
        +----------------------+
        ```
        """
        
        self.run_check()
        
        self.exec_hook("epoch_pre")
        self.init_run()
        for epoch in range(0, self.epoch_num):
            if epoch <= self.cur_epoch and self.cur_epoch != 0:
                for key, logger in self.logger_dict.items():
                    logger.info(
                        f"| epoch: [{epoch+1}] <= resume_epoch: [{ self.cur_epoch}], continue... "
                    )
                continue

            self.init_epoch(epoch)
            self.exec_hook("iter_pre")
            r_tic = time.time()
            for i, data in enumerate(self.dataloader):
                self.init_iter(r_tic=r_tic)
                self.run_one_iter(data=data, epoch=epoch)
                self.end_iter(b_tic=r_tic, step=i, epoch=epoch)
            self.exec_hook("iter_end")
            self.end_epoch(epoch=epoch)

            if epoch % self.save_interval == 0 and self.mode in ['train']:
                yield epoch

            if self.mode in ['validation'] and self.best_score > self.memory_score:
                for key, logger in self.logger_dict.items():
                    logger.log(coloring("Save the best model (" + self.criterion_metric_name + f"){int(self.best_score * 10000) / 10000}.", "OKGREEN"))
                yield epoch

        self.exec_hook("epoch_end")
        self.end_run()