'''
Author       : Thyssen Wen
Date         : 2023-10-09 17:09:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 17:10:34
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/dynamic_stream_base_dataset.py
'''
import abc
import random
import numpy as np
import math
from typing import Iterator, Dict, List

from .stream_base_dataset import StreamDataset
from svtas.utils import AbstractBuildFactory

class BaseDynamicStreamGenerator(metaclass=abc.ABCMeta):
    max_len: int
    def __init__(self,
                 max_len: int = 0) -> None:
        self.max_len = max_len
        self._precise_sliding_num = None
    
    @property
    def precise_sliding_num(self) -> List:
        return self._precise_sliding_num.tolist()
    
    @abc.abstractmethod
    def __next__(self) -> Dict:
        pass

    def update_precise_sliding_num(self, cur_len):
        if self._precise_sliding_num:
            t = self.len_batch >= cur_len
            self._precise_sliding_num += t
    
    def __iter__(self, max_len: int, len_batch: List[int] = None) -> Iterator:
        self.max_len = max_len
        self.len_batch = len_batch
        if self.len_batch:
            self.len_batch = np.array(self.len_batch)
            self._precise_sliding_num = np.zeros_like(self.len_batch)
        return self

@AbstractBuildFactory.register('dynamic_stream_generator')
class ListChoiceDynamicStreamGenerator(BaseDynamicStreamGenerator):
    def __init__(self,
                 strategy: str,
                 clip_seg_num_list: List,
                 sample_rate_list: List,
                 clip_seg_num_probability_list: List = None,
                 sample_rate_probability_list: List = None) -> None:
        super().__init__()
        assert strategy in ['random', 'loop'], f"Unsupport strategy: {strategy}!"
        if clip_seg_num_probability_list is not None:
            assert len(clip_seg_num_list) == len(clip_seg_num_probability_list), f"length of clip_seg_num_list and length of clip_seg_num_probability_list must be same!"
        if sample_rate_probability_list is not None:
            assert len(sample_rate_list) == len(sample_rate_probability_list), f"length of sample_rate_list and length of sample_rate_probability_list must be same!"
        self.strategy = strategy
        self.clip_seg_num_list = clip_seg_num_list
        self.clip_seg_num_probability_list = clip_seg_num_probability_list
        self.sample_rate_list = sample_rate_list
        self.sample_rate_probability_list = sample_rate_probability_list
    
    def _random_generator(self) -> Dict:
        cur_len = 0
        while cur_len < self.max_len:
            if self.clip_seg_num_probability_list is None:
                clip_seg_num = random.choice(self.clip_seg_num_list)
            else:
                p = np.array(self.clip_seg_num_probability_list)
                clip_seg_num = np.random.choice(self.clip_seg_num_list, p=p.ravel())
            if self.sample_rate_probability_list is None:
                sample_rate = random.choice(self.sample_rate_list)
            else:
                p = np.array(self.sample_rate_probability_list)
                sample_rate = np.random.choice(self.sample_rate_list, p=p.ravel())

            if cur_len + sample_rate * clip_seg_num < self.max_len:
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                yield dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)
            else:
                left_len = self.max_len - cur_len
                clip_seg_num = math.floor(left_len / sample_rate)
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)

    def _loop_generator(self) -> Dict:
        cur_len = 0
        clip_seg_num_idx = 0
        sample_rate_idx = 0
        while cur_len < self.max_len:
            clip_seg_num = self.clip_seg_num_list[clip_seg_num_idx]
            sample_rate = self.sample_rate_list[sample_rate_idx]

            if clip_seg_num_idx >= len(self.clip_seg_num_list):
                clip_seg_num_idx = 0
            if sample_rate_idx >= len(self.sample_rate_list):
                sample_rate_idx = 0

            if cur_len + sample_rate * clip_seg_num < self.max_len:
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                yield dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)
            else:
                left_len = self.max_len - cur_len
                clip_seg_num = math.floor(left_len / sample_rate)
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)
    
    def __next__(self) -> Dict:
        if self.strategy in ['random']:
            return self._random_generator()
        elif self.strategy in ['loop']:
            return self._loop_generator()

@AbstractBuildFactory.register('dynamic_stream_generator')
class RandomDynamicStreamGenerator(BaseDynamicStreamGenerator):
    def __init__(self,
                 clip_seg_num_range_list: List,
                 sample_rate_range_list: List) -> None:
        super().__init__()
        assert len(clip_seg_num_range_list) == 2, "length of clip_seg_num_range_list must be two and [min, max]!"
        assert len(sample_rate_range_list) == 2, "length of sample_rate_range_list must be two and [min, max]!"
        self.clip_seg_num_max = clip_seg_num_range_list[1]
        self.clip_seg_num_min = clip_seg_num_range_list[0]
        self.sample_rate_max = sample_rate_range_list[1]
        self.sample_rate_min = sample_rate_range_list[0]
    
    def __next__(self) -> Dict:
        cur_len = 0
        while cur_len < self.max_len:
            clip_seg_num = random.randint(self.clip_seg_num_min, self.clip_seg_num_max)
            sample_rate = random.randint(self.sample_rate_min, self.sample_rate_max)

            if cur_len + sample_rate * clip_seg_num < self.max_len:
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                yield dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)
            else:
                left_len = self.max_len - cur_len
                clip_seg_num = math.floor(left_len / sample_rate)
                cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=cur_len)
            
class DynamicStreamDataset(StreamDataset):
    dynamic_stream_generator: BaseDynamicStreamGenerator
    def __init__(self,
                 file_path,
                 gt_path,
                 pipeline,
                 actions_map_file_path,
                 temporal_clip_batch_size,
                 video_batch_size,
                 dynamic_stream_generator: Dict,
                 train_mode=False,
                 suffix='',
                 dataset_type='gtea',
                 data_prefix=None,
                 drap_last=False,
                 local_rank=-1,
                 nprocs=1,
                 data_path=None) -> None:
        super().__init__(file_path, gt_path, pipeline, actions_map_file_path, temporal_clip_batch_size,
                         video_batch_size, train_mode, suffix, dataset_type, data_prefix, drap_last,
                         local_rank, nprocs, data_path)
        self.dynamic_stream_generator = AbstractBuildFactory.create_factory('dynamic_stream_generator').create(dynamic_stream_generator)
    
    def _step_sliding_sampler(self, woker_id, num_workers, info_list):
        # dispatch function
        current_sliding_cnt = woker_id * self.temporal_clip_batch_size
        mini_sliding_cnt = 0
        next_step_flag = False
        for step, sliding_num, dynamic_sample_list, info in info_list:
            while current_sliding_cnt < sliding_num and len(info) > 0:
                while mini_sliding_cnt < self.temporal_clip_batch_size:
                    if current_sliding_cnt < sliding_num:
                        info.update(dynamic_sample_list[current_sliding_cnt])
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