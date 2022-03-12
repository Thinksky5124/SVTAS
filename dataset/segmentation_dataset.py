import os.path as osp
import copy
import random
import numpy as np
import os

import torch
import random
import torch.utils.data as data

from utils.config import get_logger

logger = get_logger("ETETS")

class SegmentationDataset(data.Dataset):
    """Video dataset for action recognition
       The dataset loads raw videos and apply specified transforms on them.
       The index file is a file with multiple lines, and each line indicates
       a sample video with the filepath and label, which are split with a whitesapce.
       Example of a inde file:
        file tree:
        ─── GTEA
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
                 sample_idx_list,
                 sliding_window=15,
                 clip_seg_num=15,
                 sample_rate=4,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 test_mode=False):
        super().__init__()
        self.suffix = suffix
        self.videos_path = videos_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        self.sample_idx_list = sample_idx_list
        self.sliding_window = sliding_window
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.test_mode = test_mode
        self.pipeline = pipeline
        self.info = self.load_file()
        

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

    def parse_file_paths(self, input_path):
        if self.dataset_type in ['gtea', '50salads']:
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

    def load_file(self):
        """Load index file to get video information."""
        video_segment_lists = self.parse_file_paths(self.file_path)
        info = []
        max_len = 0
        for video_segment in video_segment_lists:
            if self.dataset_type in ['gtea', '50salads']:
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
            info.append(
                dict(filename=video_path,
                     labels=classes,
                     video_name=video_name))
            if max_len < len(content):
                max_len = len(content)

        # construct sliding num
        max_len = max_len + ((self.clip_seg_num * self.sample_rate) - max_len % (self.clip_seg_num * self.sample_rate))
        sliding_num = max_len // self.sliding_window
        if max_len % self.sliding_window != 0:
            sliding_num = sliding_num + 1
        self.sliding_num = sliding_num

        results = []
        for sample_idx in self.sample_idx_list:
            results.append(info[sample_idx])
        return results

    def prepare_train(self, idx):
        """TRAIN & VALID. Prepare the data for training/valid given the index."""
        #Try to catch Exception caused by reading corrupted video file
        imgs_list = []
        labels_list = []
        for single_info in self.info:
            sample_segment = single_info
            sample_segment['sample_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            imgs_list.append(np.expand_dims(sample_segment['imgs'], axis=0))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0))

        imgs = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return imgs, labels, idx

    def prepare_test(self, idx):
        imgs_list = []
        labels_list = []
        for single_info in self.info:
            sample_segment = single_info
            sample_segment['sample_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            imgs_list.append(np.expand_dims(sample_segment['imgs'], axis=0))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0))

        imgs = np.concatenate(imgs_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        return imgs, labels, idx
    
    def __len__(self):
        """get the size of the dataset."""
        return self.sliding_num

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        if self.test_mode:
            return self.prepare_test(idx)
        else:
            return self.prepare_train(idx)

class VideoSamplerDataset(data.Dataset):
    def __init__(self,
                 file_path,
                 dataset_type):
        super().__init__()
        
        self.file_path = file_path
        self.dataset_type = dataset_type
        self.info = self.load_file()

    def parse_file_paths(self, input_path):
        if self.dataset_type in ['gtea', '50salads']:
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

    def load_file(self):
        """Load index file to get video information."""
        video_segment_lists = self.parse_file_paths(self.file_path)
        return video_segment_lists
    
    def __len__(self):
        """get the size of the dataset."""
        return len(self.info)

    def __getitem__(self, idx):
        """ Get the sample for either training or testing given index"""
        return idx