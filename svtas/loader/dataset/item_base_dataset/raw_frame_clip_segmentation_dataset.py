'''
Author       : Thyssen Wen
Date         : 2022-12-03 19:59:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-04 21:17:17
Description  : Raw Frame Clip Segmentation Dataset
FilePath     : /SVTAS/svtas/loader/dataset/item_base_dataset/raw_frame_clip_segmentation_dataset.py
'''
import torch
import copy
import os
import os.path as osp

import numpy as np
from ...builder import DATASET
from .raw_frame_segmentation_dataset import RawFrameSegmentationDataset

@DATASET.register()
class RawFrameClipSegmentationDataset(RawFrameSegmentationDataset):
    def __init__(self, videos_path, **kwargs):
        super().__init__(videos_path, **kwargs)
    
    def __getitem__(self, index):
        output_data_dict = {}
        video_segment = self.info[index]
        if self.dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            video_name = video_segment.split('.')[0]
            label_path = os.path.join(self.gt_path, video_name + '.txt')

            video_path = os.path.join(self.videos_path, video_name + '.mp4')
            if not osp.isfile(video_path):
                video_path = os.path.join(self.videos_path, video_name + '.avi')
                if not osp.isfile(video_path):
                    video_path = os.path.join(self.videos_path, video_name + '.npy')
                    if not osp.isfile(video_path):
                        raise NotImplementedError
        elif self.dataset_type in ['breakfast']:
            video_segment_name, video_segment_path = video_segment
            video_name = video_segment_name.split('.')[0]
            label_path = os.path.join(self.gt_path, video_name + '.txt')

            video_path = os.path.join(self.videos_path, video_segment_path + '.mp4')
            if not osp.isfile(video_path):
                video_path = os.path.join(self.videos_path, video_segment_path + '.avi')
                if not osp.isfile(video_path):
                    video_path = os.path.join(self.videos_path, video_name + '.npy')
                    if not osp.isfile(video_path):
                        raise NotImplementedError
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(len(content), dtype='int64')
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]

        data_dict = {}
        data_dict['filename'] = video_path
        data_dict['raw_labels'] = copy.deepcopy(classes)
        data_dict['video_name'] = video_name

        data_dict = self.pipeline(data_dict)

        output_data_dict['imgs'] = copy.deepcopy(torch.stack(data_dict['imgs'], dim=0))
        output_data_dict['labels'] = copy.deepcopy(np.stack(data_dict['labels'], axis=0).astype(np.int64))
        output_data_dict['masks'] = copy.deepcopy(np.stack(data_dict['masks'], axis=0).astype(np.float32))
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict