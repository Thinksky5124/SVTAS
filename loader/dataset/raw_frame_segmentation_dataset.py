'''
Author       : Thyssen Wen
Date         : 2022-05-26 22:37:55
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 22:47:07
Description  : Raw Frame Segmentation Dataset
FilePath     : /ETESVS/loader/dataset/raw_frame_segmentation_dataset.py
'''
import os.path as osp
import numpy as np
import os
import copy
import torch
import torch.utils.data as data
from ..builder import DATASET

@DATASET.register()
class RawFrameSegmentationDataset(data.Dataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
        file tree:
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
       Args:
           file_path(str): Path to the index file.
           pipeline(XXX): A sequence of data transforms.
           **kwargs: Keyword arguments for ```BaseDataset```.
    """
    def __init__(self,
                 file_path,
                 videos_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 clip_seg_num=15,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1):
        super().__init__()
        self.suffix = suffix
        self.videos_path = videos_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        self.clip_seg_num = clip_seg_num
        
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.pipeline = pipeline

        # distribute
        self.local_rank = local_rank
        self.nprocs = nprocs
        self.drap_last = drap_last
        if self.nprocs > 1:
            self.drap_last = True

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        
        self.info = self.load_file()
    
    def _viodeo_sample_shuffle(self):
        pass

    def load_file(self):
        """Load index file to get video information."""
        if self.dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            file_ptr = open(self.file_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
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
            info = refine_info
        return info
    
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
                    raise NotImplementedError
        elif self.dataset_type in ['breakfast']:
            video_segment_name, video_segment_path = video_segment
            video_name = video_segment_name.split('.')[0]
            label_path = os.path.join(self.gt_path, video_name + '.txt')

            video_path = os.path.join(self.videos_path, video_segment_path + '.mp4')
            if not osp.isfile(video_path):
                video_path = os.path.join(self.videos_path, video_segment_path + '.avi')
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

        output_data_dict['imgs'] = copy.deepcopy(data_dict['imgs'])
        output_data_dict['labels'] = copy.deepcopy(data_dict['labels'])
        output_data_dict['masks'] = copy.deepcopy(data_dict['masks'])
        output_data_dict['vid_list'] = video_name
        output_data_dict['sliding_num'] = 1
        output_data_dict['precise_sliding_num'] = 1.0
        output_data_dict['current_sliding_cnt'] = 0

        return output_data_dict

    def __len__(self):
        return len(self.info)
