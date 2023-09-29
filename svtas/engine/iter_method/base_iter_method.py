'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:37:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-28 21:39:27
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/base_iter_method.py
'''
import abc
import time
from typing import Any
from svtas.utils.logger import BaseLogger, BaseRecord
from svtas.model_pipline import BaseModelPipline
from svtas.loader.dataloader import BaseDataloader

class BaseIterMethod(metaclass=abc.ABCMeta):
    dataloader: BaseDataloader
    model_pipline: BaseModelPipline
    logger: BaseLogger
    record: BaseRecord

    def __init__(self, *args, **kwargs) -> None:
        pass

    def init_iter(self,
                  logger: BaseLogger,
                  record: BaseRecord,
                  model_pipline: BaseModelPipline):
        self.logger = logger
        self.record = record
        self.model_pipline = model_pipline

    def set_dataloader(self, dataloader: BaseDataloader):
        self.dataloader = dataloader
    
    def run_one_forward(self, data_dict):
        score, loss_dict = self._run_model_pipline(data_dict)
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
            
    @abc.abstractmethod
    def run(self, *args: Any, **kwds: Any) -> Any:
        raise NotImplementedError("You must implement __call__ function!")