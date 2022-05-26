'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:41:27
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 20:20:10
Description  : Collect function
FilePath     : /ETESVS/loader/pipline/collect_fn.py
'''
import re
from ..builder import PIPLINE
import torch
import copy

@PIPLINE.register()
class StreamBatchCompose():
    def __init__(self, to_tensor_keys=["imgs", "masks", "labels"]):
        self.to_tensor_keys = to_tensor_keys

    def __call__(self, batch):
        result_batch = []
        for index in range(len(batch)):
            data = {}
            for key, value in batch[index].items():
                if key in self.to_tensor_keys:
                    if not torch.is_tensor(value):
                        data[key] = torch.tensor(value)
                    else:
                        data[key] = value
                else:
                    data[key] = value
            result_batch.append(data)
        return result_batch

@PIPLINE.register()
class BatchCompose():
    def __init__(self,
                 ignore_index=-100,
                 max_keys=[""],
                 compress_keys=[""],
                 dropout_keys=[""],
                 to_tensor_keys=["imgs", "masks", "labels"]):
        self.to_tensor_keys = to_tensor_keys
        self.max_keys = max_keys
        self.compress_keys = compress_keys
        self.dropout_keys = dropout_keys
        self.ignore_index = ignore_index
    
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
                out_tensor = output_list[0]
            else:
                out_tensor = torch.nn.utils.rnn.pad_sequence(output_list, batch_first=True, padding_value=0.0)
        else:
            out_tensor = output_list[0].unsqueeze(0)
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
        return [data]