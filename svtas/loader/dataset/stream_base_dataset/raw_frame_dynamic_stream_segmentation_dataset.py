'''
Author       : Thyssen Wen
Date         : 2023-10-09 14:52:08
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 21:00:41
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

from svtas.utils import AbstractBuildFactory
from .dynamic_stream_base_dataset import DynamicStreamDataset


@AbstractBuildFactory.register('dataset')
class RawFrameDynamicStreamSegmentationDataset(DynamicStreamDataset):
    def __init__(self,
                 videos_path,
                 dynamic_stream_generator,
                 **kwargs):
        super().__init__(dynamic_stream_generator=dynamic_stream_generator, **kwargs)
        self.videos_path = videos_path

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
        imgs_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        precise_sliding_num_list = []
        
        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            imgs_list.append(copy.deepcopy(sample_segment['imgs'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['masks'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))
            precise_sliding_num_list.append(np.expand_dims(sample_segment['precise_sliding_num'], axis=0).copy())

        imgs = copy.deepcopy(torch.concat(imgs_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        precise_sliding_num = copy.deepcopy(np.concatenate(precise_sliding_num_list, axis=0).astype(np.float32))

        # compose result
        data_dict = {}
        data_dict['imgs'] = imgs
        data_dict['labels'] = labels
        data_dict['masks'] = masks
        data_dict['precise_sliding_num'] = precise_sliding_num
        data_dict['vid_list'] = vid_list
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['imgs'] = 0
        data_dict['labels'] = 0
        data_dict['masks'] = 0
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['precise_sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict

@AbstractBuildFactory.register('dataset')
class DiffusionRawFrameDynamicStreamSegmentationDataset(RawFrameDynamicStreamSegmentationDataset):
    def _get_one_videos_clip(self, idx, info):
        imgs_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        labels_onehot_list = []
        boundary_prob_list = []
        precise_sliding_num_list = []
        
        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            imgs_list.append(copy.deepcopy(sample_segment['imgs'].unsqueeze(0)))
            labels_onehot_list.append(copy.deepcopy(sample_segment['labels_onehot'].unsqueeze(0)))
            boundary_prob_list.append(copy.deepcopy(sample_segment['boundary_prob'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['masks'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))
            precise_sliding_num_list.append(np.expand_dims(sample_segment['precise_sliding_num'], axis=0).copy())

        imgs = copy.deepcopy(torch.concat(imgs_list, dim=0))
        labels_onehot = copy.deepcopy(torch.concat(labels_onehot_list, dim=0))
        boundary_prob = copy.deepcopy(torch.concat(boundary_prob_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        precise_sliding_num = copy.deepcopy(np.concatenate(precise_sliding_num_list, axis=0).astype(np.float32))

        # compose result
        data_dict = {}
        data_dict['imgs'] = imgs
        data_dict['labels_onehot'] = labels_onehot
        data_dict['boundary_prob'] = boundary_prob
        data_dict['labels'] = labels
        data_dict['masks'] = masks
        data_dict['precise_sliding_num'] = precise_sliding_num
        data_dict['vid_list'] = vid_list
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['imgs'] = 0
        data_dict['labels_onehot'] = 0
        data_dict['boundary_prob'] = 0
        data_dict['labels'] = 0
        data_dict['masks'] = 0
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['precise_sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict