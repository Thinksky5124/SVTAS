'''
Author       : Thyssen Wen
Date         : 2023-10-18 15:49:15
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 21:01:52
Description  : file content
FilePath     : /SVTAS/svtas/engine/iter_method/iter.py
'''
import time
from typing import Dict
from .epoch import EpochMethod
from svtas.utils import AbstractBuildFactory
from svtas.utils.logger import coloring

@AbstractBuildFactory.register('engine_component')
class IterMethod(EpochMethod):
    def __init__(self,
                 iter_num: int,
                 batch_size: int,
                 criterion_metric_name: str,
                 mode: str = 'train',
                 save_interval: int = 10,
                 test_interval: int = -1) -> None:
        super().__init__(1, batch_size, mode, criterion_metric_name,
                         save_interval, test_interval)
        self.iter_num = iter_num
    
    def register_epoch_pre_hook(self, func):
        """
        You should not modify value in hook function!
        """
        raise NotImplementedError("Do not use this func in IterMethod!")

    def register_epoch_end_hook(self, func):
        """
        You should not modify value in hook function!
        """
        raise NotImplementedError("Do not use this func in IterMethod!")
        
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
    
    def init_run(self):
        self.cur_iter = 0
        super().init_epoch(epoch=1)
        return super().init_run()
    
    def end_run(self):
        self.end_epoch(epoch=1)
        return super().end_run()
    
    def run(self) -> float:
        """
        run function processing
        ```
        +-----------------------+
        |  init run             |
        +-----------------------+
        |  iter pre hook        |
        +-----------------------+
        |  start iter enmu      |
        +-----------------------+
        |  init iter            |
        +-----------------------+
        |  run one bactch       |
        +-----------------------+
        |  every batch end hook |
        +-----------------------+
        |  end iter             |
        +-----------------------+
        |  end iter enmu        |
        +-----------------------+
        |  iter end hook        |
        +-----------------------+
        |  end run              |
        +-----------------------+
        ```
        """
        self.run_check()
        
        self.exec_hook("epoch_pre")
        self.init_run()
        self.exec_hook("iter_pre")
        r_tic = time.time()
        for iter_cnt, data in zip(range(0, self.iter_num), self.dataloader):
            if iter_cnt <= self.cur_iter and self.cur_iter != 0:
                for key, logger in self.logger_dict.items():
                    logger.info(
                        f"| iter: [{iter_cnt+1}] <= resume_epoch: [{ self.cur_iter}], continue... "
                    )
                continue
            
            
            self.init_iter(r_tic=r_tic)
            self.run_one_batch(data=data)
            self.end_iter(b_tic=r_tic, step=iter_cnt, epoch=1)
            r_tic = time.time()
            if iter_cnt % self.save_interval == 0 and self.mode in ['train', 'profile']:
                yield iter_cnt

            if self.mode in ['validation'] and self.best_score > self.memory_score:
                for key, logger in self.logger_dict.items():
                    logger.log(coloring("Save the best model (" + self.criterion_metric_name + f"){int(self.best_score * 10000) / 10000}.", "OKGREEN"))
                yield iter_cnt
        self.exec_hook("iter_end")
        self.end_run()
    
    def end(self):
        self.model_pipline.end_model_pipline()
        return super().end()

    def save(self) -> Dict:
        save_dict = super().save()
        if self.mode == 'train':
            save_dict['cur_iter'] = self.cur_iter
        return save_dict
    
    def load(self, load_dict: Dict) -> None:
        if self.mode == 'train':
            self.cur_iter = load_dict['cur_iter']
            self.resume_flag = True
        return super().load(load_dict)