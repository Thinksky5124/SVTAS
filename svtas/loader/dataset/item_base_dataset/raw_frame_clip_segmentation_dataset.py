'''
Author       : Thyssen Wen
Date         : 2022-12-03 19:59:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-09 21:09:29
Description  : Raw Frame Clip Segmentation Dataset
FilePath     : /SVTAS/svtas/loader/dataset/item_base_dataset/raw_frame_clip_segmentation_dataset.py
'''
import torch
import copy
import os
import os.path as osp

import numpy as np
from ...builder import DATASET
from .item_base_dataset import ItemDataset

@DATASET.register()
class RawFrameClipSegmentationDataset(ItemDataset):
    def __init__(self,
                 videos_path,
                 sliding_window=60,
                 **kwargs):
        self.sliding_window = sliding_window
        self.videos_path = videos_path
        super().__init__(**kwargs)
    
    def load_file(self):
        """Load index file to get video information."""
        if self.dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            file_ptr = open(self.file_path, 'r')
            video_info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        elif self.dataset_type in ['breakfast']:
            file_ptr = open(self.file_path, 'r')
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
            video_info = refine_info
        
        info = []
        for video_segment in video_info:
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
            vid_len = len(content)

            for start_frame in range(0, vid_len, self.sliding_window):
                video_segment_dict = dict()
                video_segment_dict['label_path'] = label_path
                video_segment_dict['video_path'] = video_path
                video_segment_dict['video_name'] = video_name
                video_segment_dict['start_frame'] = start_frame
                video_segment_dict['end_frame'] = start_frame + self.sliding_window
                info.append(video_segment_dict)
        return info
    
    def __getitem__(self, index):
        output_data_dict = {}
        video_segment_dict = self.info[index]
        
        label_path = video_segment_dict['label_path']
        video_path = video_segment_dict['video_path']
        video_name = video_segment_dict['video_name']
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]
        classes = np.zeros(len(content), dtype='int64')
        for i in range(len(content)):
            classes[i] = self.actions_dict[content[i]]

        data_dict = {}
        data_dict['filename'] = video_path
        data_dict['raw_labels'] = copy.deepcopy(classes)
        data_dict['video_name'] = video_name
        data_dict['start_frame'] = copy.deepcopy(video_segment_dict['start_frame'])
        data_dict['end_frame'] = copy.deepcopy(video_segment_dict['end_frame'])

        data_dict = self.pipeline(data_dict)

        output_data_dict['imgs'] = copy.deepcopy(data_dict['imgs'])
        output_data_dict['labels'] = copy.deepcopy(data_dict['labels'].astype(np.int64))
        output_data_dict['masks'] = copy.deepcopy(data_dict['masks'].astype(np.float32))
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict