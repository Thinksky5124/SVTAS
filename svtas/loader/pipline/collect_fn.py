'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:41:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 23:02:23
Description  : Collect function
FilePath     : /SVTAS/svtas/loader/pipline/collect_fn.py
'''
import abc
import copy
from typing import Any, Dict, List

import torch
import numpy as np
from svtas.utils import AbstractBuildFactory


class BaseCompose(metaclass=abc.ABCMeta):
    def __init__(self,
                 to_tensor_keys: List[str] = []) -> None:
        self.to_tensor_keys = to_tensor_keys
    
    @abc.abstractmethod
    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        pass

@AbstractBuildFactory.register('dataset_pipline')
class BatchCompose(BaseCompose):
    def __init__(self,
                 ignore_index=-100,
                 max_keys=[""],
                 compress_keys=["current_sliding_cnt", "current_sliding_cnt", "step"],
                 dropout_keys=[""],
                 to_tensor_keys=["imgs", "masks", "labels"],
                 clip_compress_keys=[]):
        super().__init__(to_tensor_keys)
        self.max_keys = max_keys
        self.compress_keys = compress_keys
        self.dropout_keys = dropout_keys
        self.ignore_index = ignore_index
        self.clip_compress_keys = clip_compress_keys
    
    def _compose_list(self, batch, key):
        output_list = []
        for sample in batch:
            data = sample[key]
            output_list.append(data)
        return output_list
    
    def _compose_max(self, batch, key):
        output_list = []
        for sample in batch:
            data = sample[key]
            output_list.append(data)
        return max(output_list)

    def _compose_compress(self, batch, key):
        return batch[0][key]
    
    def _compose_tensor(self, batch, key):
        output_list = []
        for sample in batch:
            data = sample[key]
            if not torch.is_tensor(data):
                data = torch.tensor(data)
            output_list.append(data)
        
        if len(batch) > 1:
            if key in ["labels"]:
                out_tensor = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True, padding_value=self.ignore_index)
            elif key in ["precise_sliding_num"]:
                out_tensor = torch.tensor(output_list)
            else:
                out_tensor = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True, padding_value=0.0)
        else:
            out_tensor = output_list[0].unsqueeze(0)
        if key in self.clip_compress_keys:
            out_tensor_shape = list(out_tensor.shape)
            del out_tensor_shape[0]
            out_tensor_shape[0] = -1
            out_tensor = out_tensor.reshape(out_tensor_shape)
        return out_tensor

    def __call__(self, batch):
        data = {}
        for key in batch[0].keys():
            if key in self.to_tensor_keys:
                data[key] = copy.deepcopy(self._compose_tensor(batch, key))
            elif key in self.max_keys:
                data[key] = copy.deepcopy(self._compose_max(batch, key))
            elif key in self.compress_keys:
                data[key] = copy.deepcopy(self._compose_compress(batch, key))
            elif key in self.dropout_keys:
                pass
            else:
                data[key] = copy.deepcopy(self._compose_list(batch, key))
        return data

@AbstractBuildFactory.register('dataset_pipline')
class BatchNumpyCompose(BatchCompose):
    def _compose_tensor(self, batch, key):
        return super()._compose_tensor(batch, key).numpy()