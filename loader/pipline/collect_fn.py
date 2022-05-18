'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:41:27
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:45:44
Description  : Collect function
FilePath     : /ETESVS/loader/pipline/collect_fn.py
'''
from ..builder import PIPLINE
import torch

@PIPLINE.register()
class BatchCompose():
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
