import copy
import os
import os.path as osp

import numpy as np
import torch

from svtas.utils import AbstractBuildFactory
from .raw_frame_stream_segmentation_dataset import \
    RawFrameStreamSegmentationDataset


@AbstractBuildFactory.register('dataset')
class RGBMVsResFrameStreamSegmentationDataset(RawFrameStreamSegmentationDataset):
    def __init__(self,
                 flows_path,
                 res_path,
                 **kwargs):
        self.flows_path = flows_path
        self.res_path = res_path
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
                        flows_path = os.path.join(self.flows_path, video_name + '.mp4')
                        res_path = os.path.join(self.res_path, video_name + '.mp4')

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
                        flows_path = os.path.join(self.flows_path, video_segment_name + '.mp4')
                        res_path = os.path.join(self.res_path, video_name + '.mp4')

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
                            flows_path=flows_path,
                            res_path=res_path,
                            raw_labels=classes,
                            video_name=video_name,
                            precise_sliding_num=precise_sliding_num))

                info_proc[proces_idx] = info

            # construct sliding num
            sliding_num = max_len // self.sliding_window
            if max_len % self.sliding_window != 0:
                sliding_num = sliding_num + 1

            # nprocs sync
            for proces_idx in range(self.nprocs):
                info_list[proces_idx].append([step, sliding_num, info_proc[proces_idx]])
        return info_list
