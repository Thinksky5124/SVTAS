'''
Author       : Thyssen Wen
Date         : 2022-10-27 16:48:57
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 18:38:08
Description  : Stream Base Dataset
FilePath     : /SVTAS/loader/dataset/stream_base_dataset/stream_base_dataset.py
'''
from abc import abstractmethod

import torch
import os.path as osp
import torch.utils.data as data


class StreamDataset(data.IterableDataset):
    def __init__(self,
                 file_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 train_mode=True,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1,
                 data_path=None):
        super().__init__()
        self.suffix = suffix
        self.data_path = data_path
        self.gt_path = gt_path
        self.actions_map_file_path = actions_map_file_path
        self.dataset_type = dataset_type
        
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
                        data_dict = self._get_one_videos_clip(current_sliding_cnt, info)
                        data_dict['sliding_num'] = sliding_num
                        data_dict['step'] = step
                        data_dict['current_sliding_cnt'] = current_sliding_cnt
                        yield data_dict
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
        yield self._get_end_videos_clip()

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
    
    @abstractmethod
    def load_file(self, sample_videos_list):
        raise NotImplementedError("You should Implement it!")
    
    @abstractmethod
    def _get_end_videos_clip(self):
        raise NotImplementedError("You should Implement it!")
    
    @abstractmethod
    def _get_one_videos_clip(self, idx, info):
        raise NotImplementedError("You should Implement it!")
        

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
    