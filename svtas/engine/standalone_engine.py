'''
Author       : Thyssen Wen
Date         : 2023-10-08 20:55:44
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-17 10:40:34
Description  : file content
FilePath     : /SVTAS/svtas/engine/standalone_engine.py
'''
from .base_engine import BaseEngine
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory
from typing import Dict, List

@AbstractBuildFactory.register('engine')
class StandaloneEngine(BaseEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 running_mode = 'train') -> None:
        super().__init__(model_name,
                         model_pipline,
                         logger_dict,
                         record,
                         metric,
                         iter_method,
                         checkpointor,
                         running_mode)

    def init_engine(self, dataloader: BaseDataloader = None):
        if dataloader is not None:
            self.set_dataloader(dataloader=dataloader)
        self.iter_method.init_iter_method(logger_dict=self.logger_dict,
                                          record=self.record,
                                          metric=self.metric,
                                          model_pipline=self.model_pipline)
        self.model_pipline.to(device=self.model_pipline.device)
        self.record.init_record()
        self.checkpointor.init_ckpt()
        # set running mode
        self.iter_method.mode = self.running_mode
        if self.running_mode == 'train':
            self.model_pipline.train()
        else:
            self.model_pipline.eval()

    def set_dataloader(self, dataloader: BaseDataloader):
        self.iter_method.set_dataloader(dataloader=dataloader)
    
    def resume_impl(self, load_dict: Dict):
        self.model_pipline.load(load_dict['model_pipline'])
        self.iter_method.load(load_dict['iter_method'])
        self.record.load(load_dict['record'])

    def resume(self, path: str = None):
        if self.checkpointor.load_flag and path is None:
            load_dict = self.checkpointor.load()
            for key, logger in self.logger_dict.items():
                logger.info(f"resume engine from checkpoint file: {self.checkpointor.load_path}")
        elif path is not None:
            load_dict = self.checkpointor.load(path)
            for key, logger in self.logger_dict.items():
                logger.info(f"resume engine from checkpoint file: {path}")
        else:
            raise FileNotFoundError("You must specify a valid path!")
        self.resume_impl(load_dict)

    def save(self, save_dict: Dict = {}, path: str = None, file_name: str = None):
        save_dict['model_pipline'] = self.model_pipline.save()
        save_dict['iter_method'] = self.iter_method.save()
        save_dict['record'] = self.record.save()
        if self.checkpointor.save_flag and path is None:
            self.checkpointor.save(save_dict = save_dict, file_name = file_name)
        elif path is not None:
            self.checkpointor.save(save_dict = save_dict, path = path, file_name = file_name)
        else:
            raise FileNotFoundError("You must specify a valid path!")

    def run(self):
        for epoch in self.iter_method.run():
            if self.running_mode in ['train']:
                self.save(file_name = self.model_name + f"_epoch_{epoch + 1:05d}")
            elif self.running_mode in ['validation']:
                self.save(file_name = self.model_name + "_best")
        
    def shutdown(self):
        self.model_pipline.end_model_pipline()
        self.iter_method.end()
        self.checkpointor.shutdown()
        for name, logger in self.logger_dict.items():
            logger.close()