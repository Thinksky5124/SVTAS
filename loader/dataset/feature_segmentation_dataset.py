'''
Author       : Thyssen Wen
Date         : 2022-05-26 15:43:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 22:30:49
Description  : feature dataset class
FilePath     : /ETESVS/loader/dataset/feature_segmentation_dataset.py
'''
import numpy as np
import os.path as osp
import os
import copy
import torch.utils.data as data
from ..builder import DATASET

@DATASET.register()
class FeatureSegmentationDataset(data.Dataset):
    def __init__(self,
                 file_path,
                 feature_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 sample_rate=4,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1) -> None:
        super().__init__()
        self.suffix = suffix
        self.feature_path = feature_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        self.sample_rate = sample_rate
        self.temporal_clip_batch_size = temporal_clip_batch_size
        
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
    
    def _viodeo_sample_shuffle(self):
        pass
    
    def load_file(self):
        """Load index file to get video information."""
        file_ptr = open(self.file_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        return info
    
    def __getitem__(self, index):
        output_data_dict = {}
        if self.dataset_type in ['gtea', '50salads', 'breakfast', 'thumos14']:
            video_segment = self.info[index]
            # load video feature
            video_name = video_segment.split('.')[0]
            label_path = os.path.join(self.gt_path, video_name + '.txt')

            feature_path = os.path.join(self.feature_path, video_name + '.npy')
            if not osp.isfile(feature_path):
                raise NotImplementedError
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(len(content), dtype='int64')
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]

        data_dict = {}
        data_dict['filename'] = feature_path
        data_dict['raw_labels'] = copy.deepcopy(classes)
        data_dict['video_name'] = video_name

        data_dict = self.pipeline(data_dict)

        output_data_dict['feature'] = copy.deepcopy(data_dict['feature'])
        output_data_dict['labels'] = copy.deepcopy(data_dict['labels'])
        output_data_dict['masks'] = copy.deepcopy(data_dict['masks'])
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict

    def __len__(self):
        return len(self.info)