'''
Author       : Thyssen Wen
Date         : 2023-10-08 20:53:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 14:41:53
Description  : file content
FilePath     : /SVTAS/svtas/engine/deepspeed_engine.py
'''
from typing import Dict
from .standalone_engine import StandaloneEngine
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine')
class DeepSpeedDistributedDataParallelEngine(StandaloneEngine):
    def init_engine(self, dataloader: BaseDataloader = None):
        if dataloader is not None:
            self.set_dataloader(dataloader=dataloader)
        self.iter_method.init_iter_method(logger_dict=self.logger_dict,
                                          record=self.record,
                                          metric=self.metric,
                                          model_pipline=self.model_pipline)
        self.model_pipline.to(device=self.model_pipline.device)
        self.record.init_record()
        self.checkpointor.init_ckpt(self.model_pipline.model)
        # set running mode
        self.iter_method.mode = self.running_mode
        if self.running_mode == 'train':
            self.model_pipline.train()
        else:
            self.model_pipline.eval()
    
    def run(self):
        for epoch in self.iter_method.run():
            if self.model_pipline.local_rank <= 0:
                if self.running_mode in ['train']:
                    self.save(file_name = self.model_name + f"_epoch_{epoch + 1:05d}")
                elif self.running_mode in ['validation']:
                    self.save(file_name = self.model_name + "_best")
    