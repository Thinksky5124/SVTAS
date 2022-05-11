'''
Author       : Thyssen Wen
Date         : 2022-05-04 20:11:18
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-11 10:48:00
Description  : file content
FilePath     : /ETESVS/dataset/rgb_flow_frame_segmentation_dataset.py
'''
import os.path as osp
import numpy as np
import os
import copy
import torch
from .raw_frame_segmentation_dataset import RawFrameSegmentationDataset
from .builder import DATASET

@DATASET.register()
class RGBFlowFrameSegmentationDataset(RawFrameSegmentationDataset):
    def __init__(self,
                 flows_path,
                 **kwargs):
        self.flows_path = flows_path
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
                                raise NotImplementedError
                        flow_path = os.path.join(self.flows_path, video_name + '.mp4')

                    elif self.dataset_type in ['breakfast']:
                        video_segment_name, video_segment_path = video_segment
                        video_name = video_segment_name.split('.')[0]
                        label_path = os.path.join(self.gt_path, video_name + '.txt')

                        video_path = os.path.join(self.videos_path, video_segment_path + '.mp4')
                        if not osp.isfile(video_path):
                            video_path = os.path.join(self.videos_path, video_segment_path + '.avi')
                            if not osp.isfile(video_path):
                                raise NotImplementedError
                        flow_path = os.path.join(self.flows_path, video_segment_name + '.mp4')

                    file_ptr = open(label_path, 'r')
                    content = file_ptr.read().split('\n')[:-1]
                    classes = np.zeros(len(content), dtype='int64')
                    for i in range(len(content)):
                        classes[i] = self.actions_dict[content[i]]

                     # caculate sliding num
                    if max_len < len(content):
                        max_len = len(content)
                    precise_sliding_num = len(content) // self.sliding_window
                    if len(content) % self.sliding_window != 0:
                        precise_sliding_num = precise_sliding_num + 1

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
            for proces_idx in range(self.nprocs):
                info_list[proces_idx].append([step, sliding_num, info_proc[proces_idx]])
        return info_list
    
    def _get_one_videos_clip(self, idx, info):
        imgs_list = []
        flow_imgs_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        precise_sliding_num_list = []

        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment['sample_sliding_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            imgs_list.append(copy.deepcopy(sample_segment['imgs'].unsqueeze(0)))
            flow_imgs_list.append(copy.deepcopy(sample_segment['flows'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['mask'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))
            precise_sliding_num_list.append(np.expand_dims(sample_segment['precise_sliding_num'], axis=0).copy())

        imgs = copy.deepcopy(torch.concat(imgs_list, dim=0))
        flow_imgs = copy.deepcopy(torch.concat(flow_imgs_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        precise_sliding_num = copy.deepcopy(np.concatenate(precise_sliding_num_list, axis=0).astype(np.float32))
        
        # compose result
        data_dict = {}
        data_dict['imgs'] = imgs
        data_dict['flows'] = flow_imgs
        data_dict['labels'] = labels
        data_dict['masks'] = masks
        data_dict['precise_sliding_num'] = precise_sliding_num
        data_dict['vid_list'] = vid_list
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['imgs'] = 0
        data_dict['flows'] = 0
        data_dict['labels'] = 0
        data_dict['masks'] = 0
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['precise_sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict