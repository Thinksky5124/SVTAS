'''
Author       : Thyssen Wen
Date         : 2023-09-28 19:42:11
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-28 19:45:06
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataloader/torch_dataloader.py
'''
from typing import Iterable, Optional, Sequence, Union
from .base_dataloader import BaseDataloader
from torch.utils.data import DataLoader, Dataset, Sampler, _collate_fn_t, _worker_init_fn_t

class TorchDataLoader(BaseDataloader, DataLoader):
    def __init__(self,
                 dataset: Dataset,
                 batch_size: int = 1,
                 shuffle: bool = None,
                 sampler: Sampler = None,
                 batch_sampler: Sampler[Sequence] = None,
                 num_workers: int = 0,
                 collate_fn: _collate_fn_t | None = None,
                 pin_memory: bool = False,
                 drop_last: bool = False,
                 timeout: float = 0,
                 worker_init_fn: _worker_init_fn_t | None = None,
                 multiprocessing_context=None,
                 generator=None, *,
                 prefetch_factor: int  = None,
                 persistent_workers: bool = False,
                 pin_memory_device: str = ""):
        super().__init__(dataset, batch_size, shuffle, sampler, batch_sampler,
                         num_workers, collate_fn, pin_memory, drop_last, timeout,
                         worker_init_fn, multiprocessing_context, generator,
                         prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                         pin_memory_device=pin_memory_device)