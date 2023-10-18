'''
Author       : Thyssen Wen
Date         : 2023-09-28 19:42:11
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:53:40
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataloader/torch_dataloader.py
'''
import os
import torch
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List
from .base_dataloader import BaseDataloader
from ..dataset import BaseDataset
from torch.utils.data import DataLoader, Dataset, Sampler
from svtas.utils import AbstractBuildFactory
from svtas.dist import get_world_size_from_os, get_rank_from_os

@AbstractBuildFactory.register('dataloader')
class TorchDataLoader(DataLoader, BaseDataloader):
    def __init__(self,
                 dataset: BaseDataset,
                 batch_size: int = 1,
                 shuffle: bool = None,
                 sampler: Sampler = None,
                 batch_sampler: Sampler[Sequence] = None,
                 num_workers: int = 0,
                 collate_fn: None = None,
                 pin_memory: bool = False,
                 timeout: float = 0,
                 worker_init_fn: None = None,
                 multiprocessing_context=None,
                 generator=None, *,
                 prefetch_factor: int  = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        if get_world_size_from_os() > 1:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=get_world_size_from_os(),
                rank=get_rank_from_os()
            )
        else:
            train_sampler = sampler
        super().__init__(dataset, batch_size, shuffle, train_sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, dataset.drop_last, timeout,
                         worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
    
    def shuffle_dataloader(self, epoch: int = 0) -> None:
        self.dataset.shuffle_dataset()
        if int(os.environ['WORLD_SIZE']) > 1:
            self.sampler.set_epoch(epoch)


@AbstractBuildFactory.register('dataloader')
class TorchStreamDataLoader(DataLoader, BaseDataloader):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = None,
                 sampler: Sampler = None,
                 batch_sampler: Sampler[Sequence] = None,
                 num_workers: int = 0,
                 collate_fn: None = None,
                 pin_memory: bool = False,
                 timeout: float = 0,
                 worker_init_fn: None = None,
                 multiprocessing_context=None,
                 generator=None, *,
                 prefetch_factor: int  = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, dataset.drop_last, timeout,
                         worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)
    
    def shuffle_dataloader(self, epoch: int = 0) -> None:
        self.dataset.shuffle_dataset()

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
                 seed: int = None,
                 iter_num: int = 1) -> None:
        super().__init__()
        self.iter_num = iter_num
        self.tensor_dict = tensor_dict
        self.random_generate = torch.Generator()
        if seed is not None:
            self.random_generate.manual_seed(seed)
    
    def generate_random_tensor(self, shape: List, dtype: str = "float32", requires_grad: bool = False):
        return torch.rand(size=shape, generator=self.random_generate, dtype=self.DATATYPE_MAP[dtype], requires_grad=requires_grad)
    
    def __next__(self) -> Dict[str, Any]:
        for i in range(self.iter_num):
            data_dict = {}
            for key, value in self.tensor_dict.items():
                if isinstance(value, dict):
                    data_dict[key] = self.generate_random_tensor(**value)
                else:
                    data_dict[key] = value
            data_dict['precise_sliding_num'] = torch.tensor([1], dtype=torch.float32)
            data_dict['vid_list'] = ["test_sample"]
            data_dict['sliding_num'] = 1
            data_dict['current_sliding_cnt'] = 0
            return data_dict

    def __iter__(self):
        return self
    
    def __len__(self):
        return self.iter_num

    def shuffle_dataloader(self, epoch) -> None:
        return super().shuffle_dataloader(epoch)