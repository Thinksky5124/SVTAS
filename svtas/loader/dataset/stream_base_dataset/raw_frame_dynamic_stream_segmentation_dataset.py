'''
Author       : Thyssen Wen
Date         : 2023-10-09 14:52:08
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 17:15:25
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/raw_frame_dynamic_stream_segmentation_dataset.py
'''
import abc
import copy
import os
import os.path as osp
import random
import numpy as np
import torch
import math
from typing import Iterator, Dict, List

from svtas.utils import AbstractBuildFactory
from .dynamic_stream_base_dataset import DynamicStreamDataset
from .raw_frame_stream_segmentation_dataset import RawFrameStreamSegmentationDataset


@AbstractBuildFactory.register('dataset')
class RawFrameDynamicStreamSegmentationDataset(DynamicStreamDataset, RawFrameStreamSegmentationDataset):
    def __init__(self,
                 videos_path,
                 dynamic_stream_generator,
                 **kwargs):
        super().__init__(videos_path, dynamic_stream_generator=dynamic_stream_generator, **kwargs)
        self.videos_path = videos_path
    
    def load_file(self, sample_videos_list):
        """Load index file to get video information."""
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
                    
                    # caculate sliding num
                    vid_len_batch.append(len(content))
                    if max_len < len(content):
                        max_len = len(content)
                    
                    info.append(
                        dict(filename=video_path,
                            raw_labels=classes,
                            video_name=video_name))
                    
                info_proc[proces_idx] = info

            # dynamic generator
            dynamic_sample_list = []
            for sample_dict in self.dynamic_stream_generator(max_len, vid_len_batch):
                dynamic_sample_list.append(sample_dict)
            # construct sliding num
            sliding_num = len(dynamic_sample_list)

            # nprocs sync
            for proces_idx in range(self.nprocs):
                info_proc[proces_idx]['precise_sliding_num'] = self.dynamic_stream_generator.precise_sliding_num[proces_idx]
                info_list[proces_idx].append([step, sliding_num, dynamic_sample_list, info_proc[proces_idx]])
        return info_list

