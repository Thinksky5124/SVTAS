'''
Author       : Thyssen Wen
Date         : 2023-09-22 16:40:18
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-28 21:40:00
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
    
    def set_test_engine(self, test_engine):
        self.test_engine = test_engine
    
    def init_epoch(self):
        self.dataloader.dataset.shuffle_dataset()
    
    def run(self, *args: Any, **kwds: Any) -> Any:
        for epoch in range(0, self.epoch_num):
            self.init_epoch()
            r_tic = time.time()
            for i, data in enumerate(self.dataloader):
                self.run_one_batch(data=data, r_tic=r_tic, epoch=epoch)
                r_tic = time.time()
    
