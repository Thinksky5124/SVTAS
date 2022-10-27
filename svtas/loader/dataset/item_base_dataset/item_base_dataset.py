'''
Author       : Thyssen Wen
Date         : 2022-10-27 16:50:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 18:37:34
Description  : Item Base Dataset
FilePath     : /SVTAS/loader/dataset/item_base_dataset/item_base_dataset.py
'''
from abc import abstractmethod

import torch
import os.path as osp
import torch.utils.data as data


class ItemDataset(data.Dataset):
    """
    ItemDataset For Temporal Video Segmentation
    Other TVS ItemDataset should inherite it.
    """
    def __init__(self,
                 file_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 data_path=None,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1):
        super().__init__()
        self.suffix = suffix
        self.data_path = data_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.pipeline = pipeline

        # distribute
        self.local_rank = local_rank
        self.nprocs = nprocs
        self.drap_last = drap_last
        if self.nprocs > 1:
            self.drap_last = True

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        
        self.info = self.load_file()

    @abstractmethod
    def _viodeo_sample_shuffle(self):
        pass

    @abstractmethod
    def load_file(self):
        raise NotImplementedError("You should Implement it!")

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError("You should Implement it!")
    
    def __len__(self):
        return len(self.info)
