'''
Author: Thyssen Wen
Date: 2022-04-27 16:13:11
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 21:03:20
Description: feature dataset class
FilePath: /ETESVS/dataset/feature_segmentation_dataset.py
'''

import numpy as np
import os.path as osp
import os
import copy
import torch
from .raw_frame_segmentation_dataset import RawFrameSegmentationDataset
from .builder import DATASET

@DATASET.register()
class FeatureSegmentationDataset(RawFrameSegmentationDataset):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def parse_file_paths(self, input_path):
        if self.dataset_type in ['gtea', '50salads', 'breakfast']:
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
            for proces_idx in range(self.nprocs):
                # convert sample
                info = []
                for video_segment in video_sample_segment_lists[proces_idx]:
                    if self.dataset_type in ['gtea', '50salads', 'breakfast']:
                        video_name = video_segment.split('.')[0]
                        label_path = os.path.join(self.gt_path, video_name + '.txt')

                        video_path = os.path.join(self.videos_path, video_name + '.npy')
                        if not osp.isfile(video_path):
                            raise NotImplementedError
                    file_ptr = open(label_path, 'r')
                    content = file_ptr.read().split('\n')[:-1]
                    classes = np.zeros(len(content), dtype='int64')
                    for i in range(len(content)):
                        classes[i] = self.actions_dict[content[i]]
                    info.append(
                        dict(filename=video_path,
                            raw_labels=classes,
                            video_name=video_name))
                    if max_len < len(content):
                        max_len = len(content)
                info_proc[proces_idx] = info

            # construct sliding num
            sliding_num = max_len // self.sliding_window
            if max_len % self.sliding_window != 0:
                sliding_num = sliding_num + 1

            # nprocs sync
            for proces_idx in range(self.nprocs):
                info_list[proces_idx].append([step, sliding_num, info_proc[proces_idx]])
        return info_list
    
    def _get_one_videos_clip(self, idx, info):
        feature_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment['sample_sliding_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            feature_list.append(copy.deepcopy(sample_segment['feature'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['mask'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))

        feature = copy.deepcopy(torch.concat(feature_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        return feature, labels, masks, vid_list
