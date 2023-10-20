'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:14:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 22:59:59
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/base_pipline.py
'''
import abc
from typing import Any, Dict
from ..wrapper import BaseModel
from svtas.dist import get_world_size_from_os, get_rank_from_os
from svtas.utils import AbstractBuildFactory

class BaseModelPipline(metaclass=abc.ABCMeta):
    model: BaseModel
    local_rank: int
    world_size: int

    def __init__(self,
                 model,
                 device) -> None:
        self._device = device
        self.model = model
        
        # prepare for distribution train
        self.local_rank = get_rank_from_os()
        self.world_size = get_world_size_from_os()

    @property
    def training(self):
        return self._training
    
    @training.setter
    def training(self, val: bool):
        self._training = val

    def train(self):
        self._training = True

    def eval(self):
        self._training = False
    
    @property
    def device(self):
        return self._device
    
    def to(self, device):
        self._device = device
    
    def set_random_seed(self, seed: int = None):
        pass

    @abc.abstractmethod
    def forward(self, data_dict):
        raise NotImplementedError("You must implement forward function!")

    @abc.abstractmethod
    def resert_model_pipline(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def end_model_pipline(self):
        pass
    
    @abc.abstractmethod
    def save(self) -> Dict:
        """
        Return model param dict readly to save
        """
        raise NotImplementedError("You must implement save_model function!")

    @abc.abstractmethod
    def load(self, param_dict: Dict) -> None:
        raise NotImplementedError("You must implement load_model function!")
    
    def load_from_ckpt_file_ckeck(self, ckpt_path: str = None):
        if ckpt_path is None:
            if self.pretrained is None:
                raise FileExistsError("Not ckpt file exits!")
            else:
                ckpt_path = self.pretrained
        return ckpt_path
    
    @abc.abstractmethod
    def load_from_ckpt_file(self, ckpt_path: str = None):
        ckpt_path = self.load_from_ckpt_file_ckeck(ckpt_path)

    @abc.abstractmethod
    def train_run(self, data_dict) -> Dict:
        pass
    
    @abc.abstractmethod
    def test_run(self, data_dict) -> Dict:
        pass

    def __call__(self, data_dict) -> Any:
        if self.training:
            return self.train_run(data_dict=data_dict)
        else:
            return self.test_run(data_dict=data_dict)

class BaseInferModelPipline(BaseModelPipline):
    def __init__(self,
            model: Dict,
            post_processing: Dict,
            device = None) -> None:
        super().__init__(model, device)
        if isinstance(model, dict):
            self.model = AbstractBuildFactory.create_factory('model').create(model)
        else:
            self.model = model
        self.post_processing = AbstractBuildFactory.create_factory('post_processing').create(post_processing)

    def to(self, device):
        pass
    
    def forward(self, data_dict):
        # move data
        outputs = self.model(data_dict)
        return outputs, data_dict
    
    def post_processing_is_init(self):
        if self.post_processing is not None:
            return self.post_processing.init_flag
        else:
            return False
        
    def init_post_processing(self, input_data) -> None:
        if self.post_processing is not None:
            vid_list = input_data['vid_list']
            sliding_num = input_data['sliding_num']
            if len(vid_list) > 0:
                self.post_processing.init_scores(sliding_num, len(vid_list))

    def update_post_processing(self, model_outputs, input_data) -> None:
        if self.post_processing is not None:
            idx = input_data['current_sliding_cnt']
            labels = input_data['labels']
            score = model_outputs['output']
            output = self.post_processing.update(score, labels, idx)
        else:
            output = None
        return output

    def output_post_processing(self, cur_vid, model_outputs = None, input_data = None):
        if self.post_processing is not None:
            # get pred result
            pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
            outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
            output_dict = dict(
                vid=cur_vid,
                outputs=outputs,
                ground_truth=ground_truth_list
            )
            return output_dict
        else:
            return {}
    
    def direct_output_post_processing(self, cur_vid, model_outputs = None, input_data = None):
        # get pred result
        output_dict = self.post_processing.output()
        return output_dict
    
    def set_post_processing_init_flag(self, val: bool):
        if self.post_processing is not None:
            self.post_processing.init_flag = val
    
    def memory_clear(self):
        self.model._clear_memory_buffer()

    def resert_model_pipline(self, *args, **kwargs):
        return super().resert_model_pipline(*args, **kwargs)
    
    def end_model_pipline(self):
        return super().end_model_pipline()
    
    def save(self) -> Dict:
        save_dict = {}
        return save_dict

    def load(self, param_dict: Dict) -> None:
        pass
    
    def train_run(self, data_dict) -> Dict:
        return self.forward(data_dict)
    
    def test_run(self, data_dict) -> Dict:
        return self.forward(data_dict)