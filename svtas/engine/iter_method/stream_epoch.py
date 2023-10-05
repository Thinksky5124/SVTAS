'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:41:13
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:17:08
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/stream_epoch.py
'''
from typing import Any
from .epoch import EpochMethod
import time
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine_component')
class StreamEpochMethod(EpochMethod):
    def __init__(self,
                 epoch_num: int,
                 logger_record_step: int = 5,
                 test_interval: int = -1) -> None:
        super().__init__(epoch_num, logger_record_step, test_interval)

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
                self.run_one_forward(sliding_seg)
    
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
                self.run_one_iter(data=data, r_tic=r_tic, epoch=epoch)
                r_tic = time.time()
                self.end_iter()
            self.exec_hook("iter_end")
            self.end_epoch()
        self.exec_hook("epoch_end")