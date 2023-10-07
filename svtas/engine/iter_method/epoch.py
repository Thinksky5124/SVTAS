'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:40:18
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 23:20:11
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/epoch.py
'''
from typing import Any, Dict
import time
from .base_iter_method import BaseIterMethod
from svtas.utils import AbstractBuildFactory
from svtas.utils.logger import coloring

@AbstractBuildFactory.register('engine_component')
class EpochMethod(BaseIterMethod):
    """
    Args:
        epoch_num: int,
        batch_size: int,
        mode: str = 'train',
        logger_iter_interval: int = 5,
        logger_epoch_interval: int = 1,
        save_interval: int = 10,
        test_interval: int = -1, without test
    """
    def __init__(self,
                 epoch_num: int,
                 batch_size: int,
                 criterion_metric_name: str,
                 mode: str = 'train',
                 logger_iter_interval: int = 5,
                 logger_epoch_interval: int = 1,
                 save_interval: int = 10,
                 test_interval: int = -1) -> None:
        super().__init__(batch_size, mode, criterion_metric_name, save_interval, test_interval)
        self.epoch_num = epoch_num
        self.logger_iter_interval = logger_iter_interval
        self.logger_epoch_interval = logger_epoch_interval
        self.cur_epoch = 0
        self.resume_flag = False
        self.current_step_vid_list = None
    
    @property
    def mode(self):
        return self._mode
    
    @mode.setter
    def mode(self, val):
        assert val in ['train', 'test', 'validation', 'infer', 'profile', 'visulaize', 'extract'], f"Unsupport mode val: {val}!"
        self._mode = val
        if self.mode in ['test', 'validation']:
            self.epoch_num = 1
            self.test_interval = -1
    
    def register_epoch_pre_hook(self, func):
        """
        You should not modify value in hook function!
        """
        self.register_hook("epoch_pre", func)

    def register_epoch_end_hook(self, func):
        """
        You should not modify value in hook function!
        """
        self.register_hook("epoch_end", func)

    def register_iter_pre_hook(self, func):
        """
        You should not modify value in hook function!
        """
        self.register_hook("iter_pre", func)

    def register_iter_end_hook(self, func):
        """
        You should not modify value in hook function!
        """
        self.register_hook("iter_end", func)
    
    def register_every_batch_end_hook(self, func):
        self.register_hook("every_batch_end", func)
    
    def set_test_engine(self, test_engine):
        self.test_engine = test_engine
    
    def init_run(self):
        super().init_run()
        self.epoch_start_time = time.time()
        self.best_epoch = 0
    
    def init_epoch(self, epoch):
        self.cur_epoch = epoch
        self.dataloader.dataset.shuffle_dataset()
        self.model_pipline.resert_model_pipline()
        self.record.init_record()
        self.model_pipline.post_processing.init_flag = False
        self.current_step_vid_list = None
    
    def end_epoch(self, epoch):
        if self.mode in ['train']:
            # report metric
            for k, v in self.metric.items():
                v.accumulate()
            # exec test if need
            if self.test_interval > 0 and epoch % self.test_interval == 0:
                test_score = self.test_hook(self.best_score)
                if test_score > self.best_score:
                    self.best_score = test_score
                    self.best_epoch = epoch

            # resume traing mode
            self.model_pipline.train()
            # update lr
            self.model_pipline.update_optim_policy()

        elif self.mode in ['validation', 'test']:
            # metric output
            Metric_dict = dict()
            for k, v in self.metric.items():
                temp_Metric_dict = v.accumulate()
                Metric_dict.update(temp_Metric_dict)
            
            if Metric_dict[self.criterion_metric_name] > self.best_score:
                self.best_score = Metric_dict[self.criterion_metric_name]

        # Computer ETA
        epoch_duration_time = time.time() - self.epoch_start_time
        if self.epoch_num > 1:
            timeArray = time.gmtime(epoch_duration_time * (self.epoch_num - (epoch + 1)))
            formatTime = f"{timeArray.tm_mday - 1} day : {timeArray.tm_hour} h : {timeArray.tm_min} m : {timeArray.tm_sec} s."
            for name, logger in self.logger_dict.items():
                logger.log(coloring(f"ETA: {formatTime}", 'OKBLUE'))
        self.epoch_start_time = time.time()

        # logger
        if epoch % self.logger_epoch_interval == 0:
            self.logger_epoch(epoch)

    def init_iter(self, r_tic):
        self.record['reader_time'].update(time.time() - r_tic)

    def end_iter(self, b_tic, step, epoch):
        self.record['batch_time'].update(time.time() - b_tic)
        if self.mode == 'train':
            self.record['lr'].update(self.model_pipline.optimizer.state_dict()['param_groups'][0]['lr'], self.batch_size)
        
        if step % self.logger_iter_interval == 0 and self.mode in ['train', 'test', 'validation']:
            self.logger_iter(step, epoch)

        if self.mode in ['infer', 'extract']:
            self.record.accumulate_record()
            for name, logger in self.logger_dict.items():
                logger.info("Step: " + str(step) + ", finish ectracting video: "+ ",".join(self.current_step_vid_list))
    
    def end_run(self):
        super().end_run()
        if self.mode == 'train':
            for key , logger in self.logger_dict.items():
                logger.info(coloring(f"The best performance on {self.criterion_metric_name} is {int(self.best_score * 10000) / 10000}, in epoch {self.best_epoch + 1}.", "PURPLE"))

    def logger_iter(self, step, epoch):
        # do log
        self.record.accumulate_record()
        ips = "ips: {:.5f} instance/sec.".format(self.batch_size / self.record["batch_time"].val)
        for name, logger in self.logger_dict.items():
            logger.log_batch(self.record, step, self.mode, ips, epoch + 1, self.epoch_num)

    def logger_epoch(self, epoch):
        ips = "avg_ips: {:.5f} instance/sec.".format(
            self.batch_size * self.record["batch_time"].count /
            (self.record["batch_time"].sum + 1e-10))
        for name, logger in self.logger_dict.items():
            logger.log_epoch(self.record, epoch + 1, self.mode, ips)

    def batch_end_step(self, input_data, outputs):
        # post processing
        if not self.model_pipline.post_processing.init_flag:
            vid_list = input_data['vid_list']
            self.current_step_vid_list = vid_list
            self.model_pipline.init_post_processing(input_data=input_data)
            if self.mode in ['infer', 'extract']:
                for name, logger in self.logger_dict.items():
                    logger.info("Current process video: " + ",".join(self.current_step_vid_list))
        self.model_pipline.update_post_processing(model_outputs=outputs, input_data=input_data)
        output_dict = self.model_pipline.output_post_processing(self.current_step_vid_list)
       
        # exec hook
        self.exec_hook('every_batch_end', output_dict, self.current_step_vid_list)
        
        # update metric
        for k, v in self.metric.items():
            acc = v.update(output_dict['vid'], output_dict['ground_truth'], output_dict['outputs'])
        if self.mode in ['train', 'test', 'validation']:
            self.record['Acc'].update(acc, len(input_data['vid_list']))

        # init post processing
        self.model_pipline.init_post_processing(input_data=input_data)
        vid_list = input_data['vid_list']
        self.current_step_vid_list = vid_list

    def run_one_batch(self, data):
        # videos batch train

        for sliding_seg in data:
            # run one batch
            outputs, loss_dict = self.run_one_forward(sliding_seg)
            self.batch_end_step(input_data=sliding_seg, outputs=outputs)
            self.record.update_record(loss_dict)

    def run(self) -> float:
        """
        run function processing
        ```
        +-----------------------+
        |  init run             |
        +-----------------------+
        |  epoch pre hook       |
        +-----------------------+
        |  start epoch enmu     |
        +-----------------------+
        |  init epoch           |
        +-----------------------+
        |  iter pre hook        |
        +-----------------------+
        |  start iter enmu      |
        +-----------------------+
        |  init iter            |
        +-----------------------+
        |  run one bactch       |
        +-----------------------+
        |  every batch end hook  |
        +-----------------------+
        |  end iter             |
        +-----------------------+
        |  end iter enmu        |
        +-----------------------+
        |  iter end hook        |
        +-----------------------+
        |  end epoch            |
        +-----------------------+
        |  epoch end hook       |
        +-----------------------+
        |  end run              |
        +-----------------------+
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
                self.run_one_batch(data=data)
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
    
    def end(self):
        self.model_pipline.end_model_pipline()
        return super().end()

    def save(self) -> Dict:
        save_dict = super().save()
        if self.mode == 'train':
            save_dict['cur_epoch'] = self.cur_epoch
        return save_dict
    
    def load(self, load_dict: Dict) -> None:
        if self.mode == 'train':
            self.cur_epoch = load_dict['cur_epoch']
            self.resume_flag = True
        return super().load(load_dict)