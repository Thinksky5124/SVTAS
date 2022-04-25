'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors: Thyssen Wen
LastEditTime: 2022-04-25 21:57:52
Description: dataset class
FilePath: /ETESVS/dataset/segmentation_dataset.py
'''
import os.path as osp
import numpy as np
import os
import copy
import torch
import torch.utils.data as data

class SegmentationDataset(data.IterableDataset):
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
                 sliding_window=60,
                 clip_seg_num=15,
                 sample_rate=4,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 train_mode=True,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1):
        super().__init__()
        self.suffix = suffix
        self.videos_path = videos_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        self.sliding_window = sliding_window
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        
        self.file_path = file_path
        self.data_prefix = osp.realpath(data_prefix) if \
            data_prefix is not None and osp.isdir(data_prefix) else data_prefix
        self.pipeline = pipeline

        # distribute
        self.local_rank = local_rank
        self.nprocs = nprocs
        self.drap_last = drap_last
        if self.nprocs > 1 and train_mode is True:
            self.drap_last = True
        self.video_batch_size = video_batch_size

        # actions dict generate
        file_ptr = open(self.actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])

        # construct sampler
        self.video_sampler_dataloader = torch.utils.data.DataLoader(
                VideoSamplerDataset(file_path=file_path),
                                    batch_size=video_batch_size,
                                    num_workers=0,
                                    shuffle=train_mode)
        if train_mode == False:
            self._viodeo_sample_shuffle()
        
        # iterable
        self.temporal_clip_batch_size = temporal_clip_batch_size

    def _viodeo_sample_shuffle(self):
        # sampler video order
        self.info_list = self._stream_order_sample(self.video_sampler_dataloader)

    def _stream_order_sample(self, video_sampler_dataloader):
        sample_videos_list = []
        self.step_num = len(video_sampler_dataloader)
        for step, sample_videos in enumerate(video_sampler_dataloader):
            if self.drap_last is True and len(list(sample_videos)) < self.video_batch_size:
                break
            sample_videos_list.append([step, list(sample_videos)])

        info_list = self.load_file(sample_videos_list).copy()
        return info_list

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
        imgs_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment['sample_sliding_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            imgs_list.append(copy.deepcopy(sample_segment['imgs'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['mask'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))

        imgs = copy.deepcopy(torch.concat(imgs_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        return imgs, labels, masks, vid_list
    
    def _genrate_sampler(self, woker_id, num_workers):
        if self.local_rank < 0:
            # single gpu train
            return self._step_sliding_sampler(woker_id=woker_id, num_workers=num_workers, info_list=self.info_list[0])
        else:
            # multi gpu train
            sample_info_list = self.info_list[self.local_rank]
            return self._step_sliding_sampler(woker_id=woker_id, num_workers=num_workers, info_list=sample_info_list)
    
    def _step_sliding_sampler(self, woker_id, num_workers, info_list):
        # dispatch function
        current_sliding_cnt = woker_id * self.temporal_clip_batch_size
        mini_sliding_cnt = 0
        next_step_flag = False
        for step, sliding_num, info in info_list:
            while current_sliding_cnt < sliding_num and len(info) > 0:
                while mini_sliding_cnt < self.temporal_clip_batch_size:
                    if current_sliding_cnt < sliding_num:
                        imgs, labels, masks, vid_list = self._get_one_videos_clip(current_sliding_cnt, info)
                        yield imgs, labels, masks, vid_list, sliding_num, step, current_sliding_cnt
                        current_sliding_cnt = current_sliding_cnt + 1
                        mini_sliding_cnt = mini_sliding_cnt + 1
                    else:
                        next_step_flag = True
                        break
                if current_sliding_cnt <= sliding_num and next_step_flag == False:
                    current_sliding_cnt = current_sliding_cnt + (num_workers - 1) * self.temporal_clip_batch_size

                if mini_sliding_cnt >= self.temporal_clip_batch_size:
                    mini_sliding_cnt = 0

            # modify num_worker
            current_sliding_cnt = current_sliding_cnt - sliding_num
            next_step_flag = False
        yield 0, 0, 0, [], 0, self.step_num, -1

    def __len__(self):
        """get the size of the dataset."""
        return self.step_num
    
    def __iter__(self):
        """ Get the sample for either training or testing given index"""
        worker_info = torch.utils.data.get_worker_info()
        # single worker
        if worker_info is None:
            sample_iterator = self._genrate_sampler(0, 1)
        else: # multiple workers
            woker_id = worker_info.id
            num_workers = int(worker_info.num_workers)
            sample_iterator = self._genrate_sampler(woker_id, num_workers)
        return sample_iterator

class VideoSamplerDataset(data.Dataset):
    def __init__(self,
                 file_path):
        super().__init__()
        
        self.file_path = file_path
        self.info = self.load_file()

    def parse_file_paths(self, input_path):
        file_ptr = open(input_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
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