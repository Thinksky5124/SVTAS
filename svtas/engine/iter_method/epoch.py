'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:40:18
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-30 10:08:21
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/epoch.py
'''
from typing import Any
import time
from .base_iter_method import BaseIterMethod
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class EpochMethod(BaseIterMethod):
    def __init__(self,
                 epoch_num: int,
                 logger_record_step: int,
                 test_interval: int = -1) -> None:
        super().__init__()
        self.epoch_num = epoch_num
        self.logger_record_step = logger_record_step
        self.test_interval = test_interval
    
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
    
    def set_test_engine(self, test_engine):
        self.test_engine = test_engine
    
    def init_epoch(self):
        self.dataloader.dataset.shuffle_dataset()
    
    def end_epoch(self):
        self.logger_epoch()

    def init_iter(self):
        pass

    def end_iter(self):
        self.logger_iter()

    def logger_iter(self):
        pass

    def logger_epoch(self):
        pass

    def run_one_batch(self, data, r_tic=None, epoch=None):
        # videos batch train
        self.record_dict['reader_time'].update(time.time() - r_tic)

        for sliding_seg in data:
            step = self.current_step
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']

            # run one batch
            self.run_one_forward(sliding_seg)
            self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)
            self.current_step = self.current_step + 1
            self.post_processing.init_flag = False
    
    def run(self, *args: Any, **kwds: Any) -> Any:
        """
        run function processing
        ```
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
        ```
        """
        super().run()

        self.exec_hook("epoch_pre")
        for epoch in range(0, self.epoch_num):
            self.init_epoch()
            r_tic = time.time()
            self.exec_hook("iter_pre")
            for i, data in enumerate(self.dataloader):
                self.init_iter()
                self.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
                r_tic = time.time()
                self.end_iter()
            self.exec_hook("iter_end")
            self.end_epoch()
        self.exec_hook("epoch_end")
    
