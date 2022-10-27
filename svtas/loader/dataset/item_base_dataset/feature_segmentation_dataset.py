'''
Author       : Thyssen Wen
Date         : 2022-05-26 15:43:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 18:31:08
Description  : feature dataset class
FilePath     : /SVTAS/loader/dataset/item_base_dataset/feature_segmentation_dataset.py
'''
import copy
import os
import os.path as osp

import numpy as np
import torch.utils.data as data

from ...builder import DATASET
from .item_base_dataset import ItemDataset


@DATASET.register()
class FeatureSegmentationDataset(ItemDataset):
    def __init__(self,
                 feature_path,
                 flow_feature_path=None,
                 **kwargs,
                 ) -> None:
        super().__init__(**kwargs)
        self.flow_feature_path = flow_feature_path
        self.feature_path = feature_path
    
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
            if self.flow_feature_path is not None:
                flow_feature_path = os.path.join(self.flow_feature_path, video_name + '.npy')
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(len(content), dtype='int64')
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]

        data_dict = {}
        data_dict['filename'] = feature_path
        if self.flow_feature_path is not None:
            data_dict['flow_feature_name'] = flow_feature_path
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
