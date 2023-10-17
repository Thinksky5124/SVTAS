'''
Author       : Thyssen Wen
Date         : 2023-09-28 19:42:11
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 09:39:41
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataloader/torch_dataloader.py
'''
import os
import torch
from typing import Iterable, Optional, Sequence, Union
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