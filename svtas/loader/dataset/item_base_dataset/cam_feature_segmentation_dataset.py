'''
Author       : Thyssen Wen
Date         : 2022-12-24 16:02:55
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-24 16:05:07
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataset/item_base_dataset/cam_feature_segmentation_dataset.py
'''
import copy
import os
import os.path as osp

import numpy as np

from ...builder import DATASET
from .feature_segmentation_dataset import \
    FeatureSegmentationDataset

@DATASET.register()
class CAMFeatureSegmentationDataset(FeatureSegmentationDataset):
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
        output_data_dict['raw_feature'] = copy.deepcopy(data_dict['raw_feature'])
        output_data_dict['labels'] = copy.deepcopy(data_dict['labels'])
        output_data_dict['masks'] = copy.deepcopy(data_dict['masks'])
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict