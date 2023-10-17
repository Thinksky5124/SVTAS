'''
Author       : Thyssen Wen
Date         : 2022-12-03 21:24:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:46:37
Description  : feature clip dataset class
FilePath     : /SVTAS/svtas/loader/dataset/item_base_dataset/feature_clip_segmentation_dataset.py
'''
import torch
import copy
import os
import os.path as osp

import numpy as np
import torch.utils.data as data

from svtas.utils import AbstractBuildFactory
from .item_base_dataset import ItemDataset


@AbstractBuildFactory.register('dataset')
class FeatureClipSegmentationDataset(ItemDataset):
    def __init__(self, feature_path, flow_feature_path=None, sliding_window=60, **kwargs) -> None:
        self.flow_feature_path = flow_feature_path
        self.feature_path = feature_path
        self.sliding_window = sliding_window
        super().__init__(**kwargs)
    
    def load_file(self):
        """Load index file to get video information."""
        file_ptr = open(self.file_path, 'r')
        video_info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()

        info = []
        for video_segment in video_info:
            if self.dataset_type in ['gtea', '50salads', 'breakfast', 'thumos14']:
                # load video feature
                video_name = video_segment.split('.')[0]
                label_path = os.path.join(self.gt_path, video_name + '.txt')

                feature_path = os.path.join(self.feature_path, video_name + '.npy')
                if not osp.isfile(feature_path):
                    raise NotImplementedError
                if self.flow_feature_path is not None:
                    flow_feature_path = os.path.join(self.flow_feature_path, video_name + '.npy')
                else:
                    flow_feature_path = None
            file_ptr = open(label_path, 'r')
            content = file_ptr.read().split('\n')[:-1]
            vid_len = len(content)
            for start_frame in range(0, vid_len, self.sliding_window):
                video_segment_dict = dict()
                video_segment_dict['label_path'] = label_path
                video_segment_dict['feature_path'] = feature_path
                video_segment_dict['flow_feature_path'] = flow_feature_path
                video_segment_dict['video_name'] = video_name
                video_segment_dict['start_frame'] = start_frame
                video_segment_dict['end_frame'] = start_frame + self.sliding_window
                info.append(video_segment_dict)
        return info
        
    def __getitem__(self, index):
        output_data_dict = {}
        video_segment_dict = self.info[index]
        
        label_path = video_segment_dict['label_path']
        feature_path = video_segment_dict['feature_path']
        flow_feature_path = video_segment_dict['flow_feature_path']
        video_name = video_segment_dict['video_name']
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(len(content), dtype='int64')
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]

        data_dict = {}
        data_dict['filename'] = feature_path
        if flow_feature_path is not None:
            data_dict['flow_feature_name'] = flow_feature_path
        data_dict['raw_labels'] = copy.deepcopy(classes)
        data_dict['video_name'] = video_name
        data_dict['start_frame'] = copy.deepcopy(video_segment_dict['start_frame'])
        data_dict['end_frame'] = copy.deepcopy(video_segment_dict['end_frame'])

        data_dict = self.pipeline(data_dict)

        output_data_dict.update(data_dict)
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict
