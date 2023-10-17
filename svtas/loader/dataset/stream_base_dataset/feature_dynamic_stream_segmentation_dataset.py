'''
Author       : Thyssen Wen
Date         : 2023-10-09 17:05:38
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:40:31
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/feature_dynamic_stream_segmentation_dataset.py
'''
import copy
import os
import os.path as osp

import numpy as np
import torch
from typing import Iterator, Dict, List

from svtas.utils import AbstractBuildFactory
from .dynamic_stream_base_dataset import DynamicStreamDataset

@AbstractBuildFactory.register('dataset')
class FeatureDynamicStreamSegmentationDataset(DynamicStreamDataset):
    def __init__(self,
                 feature_path,
                 dynamic_stream_generator: Dict,
                 flow_feature_path=None,
                 **kwargs):
        self.flow_feature_path = flow_feature_path
        self.feature_path = feature_path
        super().__init__(dynamic_stream_generator=dynamic_stream_generator, **kwargs)
    
    def parse_file_paths(self, input_path):
        if self.dataset_type in ['gtea', '50salads', 'breakfast', 'thumos14']:
            file_ptr = open(input_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        return info
    
    def load_file(self, sample_videos_list):
        """Load index file to get video feature information."""
        video_segment_lists = self.parse_file_paths(self.file_path)
        info_list = [[] for i in range(self.nprocs)]
        # sample step
        for step, sample_idx_list in sample_videos_list:
            # sample step clip
            video_sample_segment_lists = [[] for i in range(self.nprocs)]
            for sample_idx_list_idx in range(len(sample_idx_list)):
                nproces_idx = sample_idx_list_idx % self.nprocs
                sample_idx = sample_idx_list[sample_idx_list_idx]
                video_sample_segment_lists[nproces_idx].append(video_segment_lists[sample_idx])

            max_len = 0
            info_proc = [[] for i in range(self.nprocs)]
            vid_len_batch = []
            for proces_idx in range(self.nprocs):
                # convert sample
                info = []
                for video_segment in video_sample_segment_lists[proces_idx]:
                    if self.dataset_type in ['gtea', '50salads', 'breakfast', 'thumos14']:
                        video_name = video_segment.split('.')[0]
                        label_path = os.path.join(self.gt_path, video_name + '.txt')

                        video_path = os.path.join(self.feature_path, video_name + '.npy')
                        if not osp.isfile(video_path):
                            raise NotImplementedError
                    file_ptr = open(label_path, 'r')
                    content = file_ptr.read().split('\n')[:-1]
                    classes = np.zeros(len(content), dtype='int64')
                    for i in range(len(content)):
                        classes[i] = self.actions_dict[content[i]]

                    # caculate sliding num
                    vid_len_batch.append(len(content))
                    if max_len < len(content):
                        max_len = len(content)

                    if self.flow_feature_path is not None:
                        flow_feature_path = os.path.join(self.flow_feature_path, video_name + '.npy')
                        info.append(
                            dict(filename=video_path,
                                flow_feature_name=flow_feature_path,
                                raw_labels=classes,
                                video_name=video_name))
                    else:
                        info.append(
                            dict(filename=video_path,
                                raw_labels=classes,
                                video_name=video_name))
                        
                info_proc[proces_idx] = info

            # dynamic generator
            dynamic_sample_list = []
            self.dynamic_stream_generator.set_start_args(max_len, vid_len_batch)
            for sample_dict in self.dynamic_stream_generator:
                dynamic_sample_list.append(sample_dict)
            # construct sliding num
            sliding_num = len(dynamic_sample_list)

            # nprocs sync
            for proces_idx in range(self.nprocs):
                for info in info_proc[proces_idx]:
                    info['precise_sliding_num'] = self.dynamic_stream_generator.precise_sliding_num[proces_idx]
                info_list[proces_idx].append([step, sliding_num, dynamic_sample_list, info_proc[proces_idx]])
        return info_list
    
    def _get_one_videos_clip(self, idx, info):
        sample_segment = info.copy()
        sample_segment = self.pipeline(sample_segment)
        # compose result
        data_dict = {}
        data_dict.update(sample_segment)
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict
