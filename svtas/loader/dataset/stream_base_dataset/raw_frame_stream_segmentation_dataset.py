'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 20:44:11
Description: dataset class
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/raw_frame_stream_segmentation_dataset.py
'''
import copy
import os
import os.path as osp

import numpy as np
import torch

from svtas.utils import AbstractBuildFactory
from .stream_base_dataset import StreamDataset


@AbstractBuildFactory.register('dataset')
class RawFrameStreamSegmentationDataset(StreamDataset):
    """Video dataset for action recognition
        The dataset loads raw videos and apply specified transforms on them.
        The index file is a file with multiple lines, and each line indicates
        a sample video with the filepath and label, which are split with a whitesapce.
        Example of a inde file:
        file tree:

        ```
        ─── gtea
            ├── Videos
            │   ├── S1_Cheese_C1.mp4
            │   ├── S1_Coffee_C1.mp4
            │   ├── S1_CofHoney_C1.mp4
            │   └── ...
            ├── groundTruth
            │   ├── S1_Cheese_C1.txt
            │   ├── S1_Coffee_C1.txt
            │   ├── S1_CofHoney_C1.txt
            │   └── ...
            ├── splits
            │   ├── test.split1.bundle
            │   ├── test.split2.bundle
            │   ├── test.split3.bundle
            │   └── ...
            └── mapping.txt
        ```
        
        Args:
            file_path(str): Path to the index file.
            pipeline(XXX): A sequence of data transforms.
            **kwargs: Keyword arguments for ```BaseDataset```.
    """
    def __init__(self,
                 videos_path,
                 sliding_window=60,
                 need_precise_grad_accumulate=True,
                 **kwargs):
        self.videos_path = videos_path
        self.sliding_window = sliding_window
        self.need_precise_grad_accumulate = need_precise_grad_accumulate
        super().__init__(**kwargs)

    def parse_file_paths(self, input_path):
        if self.dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            file_ptr = open(input_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        elif self.dataset_type in ['breakfast']:
            file_ptr = open(input_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            refine_info = []
            for info_name in info:
                video_ptr = info_name.split('.')[0].split('_')
                file_name = ''
                for j in range(2):
                    if video_ptr[j] == 'stereo01':
                        video_ptr[j] = 'stereo'
                    file_name = file_name + video_ptr[j] + '/'
                file_name = file_name + video_ptr[2] + '_' + video_ptr[3]
                if 'stereo' in file_name:
                    file_name = file_name + '_ch0'
                refine_info.append([info_name, file_name])
            info = refine_info
        return info

    def load_file(self, sample_videos_list):
        """Load index file to get video information."""
        # Todo: accelerate this will dist, only sample and process that need by itself
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
            proces_idx = self.local_rank
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
                if max_len < len(content):
                    max_len = len(content)
                if self.need_precise_grad_accumulate:
                    precise_sliding_num = len(content) // self.sliding_window
                    if len(content) % self.sliding_window != 0:
                        precise_sliding_num = precise_sliding_num + 1
                else:
                    precise_sliding_num = 1

                info.append(
                    dict(filename=video_path,
                        raw_labels=classes,
                        video_name=video_name,
                        precise_sliding_num=precise_sliding_num))
                    
            info_proc[proces_idx] = info

            # construct sliding num
            sliding_num = max_len // self.sliding_window
            if max_len % self.sliding_window != 0:
                sliding_num = sliding_num + 1

            # nprocs sync
            info_list[proces_idx].append([step, sliding_num, info_proc[proces_idx]])
        return info_list

    def _get_one_videos_clip(self, idx, info):
        sample_segment = info.copy()
        sample_segment['sample_sliding_idx'] = idx
        sample_segment = self.pipeline(sample_segment)
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
