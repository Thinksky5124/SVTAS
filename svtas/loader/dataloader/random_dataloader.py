'''
Author       : Thyssen Wen
Date         : 2023-10-20 16:44:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-12-12 15:58:26
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataloader/random_dataloader.py
'''
import os
import torch
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List
import numpy as np
from .base_dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('dataloader')
class RandomTensorTorchDataloader(BaseDataloader):
    """
    Generate RandomTensor Torch Dataloader

    Args:
        tensor_dict: Dict[str, Dict[str, int]]
    
    Examples:
    ```
    RandomTensorTorchDataloader(tensor_dict=dict(
        imgs = dict(
            shape = [1, 3, 224, 224],
            dtype = "float32",
            requires_grad = False
        ),
        precise_sliding_num = 8
    ))
    ```
    """
    DATATYPE_MAP = {
        "float16": torch.float16,
        "float32": torch.float32,
        "flaot64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
    }
    def __init__(self,
                 tensor_dict: Dict[str, Dict[str, int]],
                 is_train: bool = True,
                 seed: int = None,
                 iter_num: int = 1) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.is_train = is_train
        self.tensor_dict = tensor_dict
        self.random_generate = torch.Generator()
        self.iter_cnt = 0
        if seed is not None:
            self.random_generate.manual_seed(seed)
    
    def generate_random_tensor(self, shape: List, dtype: str = "float32", requires_grad: bool = False):
        if dtype.startswith("int"):
            return torch.ones(size=shape, dtype=self.DATATYPE_MAP[dtype], requires_grad=requires_grad)
        else:
            return torch.rand(size=shape, generator=self.random_generate, dtype=self.DATATYPE_MAP[dtype], requires_grad=requires_grad)
    
    def __next__(self) -> Dict[str, Any]:
        if self.iter_cnt < self.iter_num:
            data_dict = {}
            for key, value in self.tensor_dict.items():
                if isinstance(value, dict):
                    data_dict[key] = self.generate_random_tensor(**value)
                else:
                    data_dict[key] = value
            if self.is_train:
                data_dict['precise_sliding_num'] = torch.tensor([1], dtype=torch.float32)
                data_dict['vid_list'] = ["test_sample"]
                data_dict['sliding_num'] = 1
                data_dict['current_sliding_cnt'] = 0
            self.iter_cnt += 1
            return data_dict
        else:
            raise StopIteration
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.iter_num

    def shuffle_dataloader(self, epoch) -> None:
        return super().shuffle_dataloader(epoch)

@AbstractBuildFactory.register('dataloader')
class RandomTensorNumpyDataloader(BaseDataloader):
    """
    Generate RandomTensor Torch Dataloader

    Args:
        tensor_dict: Dict[str, Dict[str, int]]
    
    Examples:
    ```
    RandomTensorNumpyDataloader(tensor_dict=dict(
        imgs = dict(
            shape = [1, 3, 224, 224],
            dtype = "float32",
            requires_grad = False
        ),
        precise_sliding_num = 8
    ))
    ```
    """
    DATATYPE_MAP = {
        "float16": np.float16,
        "float32": np.float32,
        "flaot64": np.float64,
        "int64": np.int64,
        "int32": np.int32,
        "int16": np.int16,
        "int8": np.int8,
    }

    def __init__(self,
                 tensor_dict: Dict[str, Dict[str, int]],
                 is_train: bool = True,
                 seed: int = None,
                 iter_num: int = 1) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.is_train = is_train
        self.tensor_dict = tensor_dict
        self.random_generate = np.random.default_rng(seed=seed)
        self.iter_cnt = 0
        if seed is not None:
            np.random.default_rng(seed=seed)
    
    def generate_random_tensor(self, shape: List, dtype: str = "float32", requires_grad: bool = False):
        return self.random_generate.random(shape).astype(self.DATATYPE_MAP[dtype])
    
    def __next__(self) -> Dict[str, Any]:
        if self.iter_cnt < self.iter_num:
            data_dict = {}
            for key, value in self.tensor_dict.items():
                if isinstance(value, dict):
                    data_dict[key] = self.generate_random_tensor(**value)
                else:
                    data_dict[key] = value
            if self.is_train:
                data_dict['precise_sliding_num'] = np.array([1], dtype=np.float32)
                data_dict['vid_list'] = ["test_sample"]
                data_dict['sliding_num'] = 1
                data_dict['current_sliding_cnt'] = 0
            self.iter_cnt += 1
            return data_dict
        else:
            raise StopIteration
        
    def __iter__(self):
        return self
    
    def __len__(self):
        return self.iter_num

    def shuffle_dataloader(self, epoch) -> None:
        return super().shuffle_dataloader(epoch)