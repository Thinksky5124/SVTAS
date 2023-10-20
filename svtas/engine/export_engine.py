'''
Author       : Thyssen Wen
Date         : 2023-10-19 15:54:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 17:08:25
Description  : file content
FilePath     : /SVTAS/svtas/engine/export_engine.py
'''
from .base_engine import BaseEngine
from svtas.loader.dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory
from svtas.inference import BaseModelConvertor
from typing import Dict, List

@AbstractBuildFactory.register('engine')
class ExportModelEngine(BaseEngine):
    dataloader: BaseDataloader
    convertor: BaseModelConvertor
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 checkpointor: Dict,
                 convertor: Dict = None,
                 running_mode='export') -> None:
        super().__init__(model_name, model_pipline, logger_dict, record,
                         None, None, checkpointor, running_mode)
        self.convertor = AbstractBuildFactory.create_factory('model_convertor').create(convertor)
    
    def init_engine(self, dataloader: BaseDataloader = None):
        if dataloader is not None:
            self.set_dataloader(dataloader=dataloader)
        self.record.init_record()
        self.checkpointor.init_ckpt()
        self.convertor.init_convertor()
        # set running mode
        self.model_pipline.eval()
    
    def set_dataloader(self, dataloader: BaseDataloader):
        self.dataloader = dataloader
    
    def resume_impl(self, load_dict: Dict):
        self.model_pipline.load(load_dict['model_pipline'])
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
        save_dict['record'] = self.record.save()
        if self.checkpointor.save_flag and path is None:
            self.checkpointor.save(save_dict = save_dict, file_name = file_name)
        elif path is not None:
            self.checkpointor.save(save_dict = save_dict, path = path, file_name = file_name)
        else:
            raise FileNotFoundError("You must specify a valid path!")
    
    def run(self):
        iter_dataloader = iter(self.dataloader)
        data = next(iter_dataloader)
        self.convertor.export(self.model_pipline.model, data=data, file_name=self.model_name + '.onnx')
        
    def shutdown(self):
        self.model_pipline.end_model_pipline()
        self.checkpointor.shutdown()
        self.convertor.shutdown()
        for name, logger in self.logger_dict.items():
            logger.close()