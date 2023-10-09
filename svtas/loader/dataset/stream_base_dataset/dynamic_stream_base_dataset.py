'''
Author       : Thyssen Wen
Date         : 2023-10-09 17:09:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 21:13:21
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
        self.init_flag = False
        self.cur_len = 0
    
    @property
    def precise_sliding_num(self) -> List:
        return self._precise_sliding_num.tolist()
    
    @abc.abstractmethod
    def __next__(self) -> Dict:
        assert self.init_flag, "Must call `set_start_args` before iter"
    
    def set_start_args(self, max_len: int, len_batch: List[int] = None):
        self.max_len = max_len
        self.len_batch = len_batch
        if self.len_batch:
            self.len_batch = np.array(self.len_batch)
            self._precise_sliding_num = np.zeros_like(self.len_batch)
        self.init_flag = True
        self.cur_len = 0

    def update_precise_sliding_num(self, cur_len):
        if self._precise_sliding_num is not None:
            t = self.len_batch >= cur_len
            self._precise_sliding_num += t
    
    def __iter__(self) -> Iterator:
        return self

@AbstractBuildFactory.register('dynamic_stream_generator')
class ListRandomChoiceDynamicStreamGenerator(BaseDynamicStreamGenerator):
    def __init__(self,
                 clip_seg_num_list: List,
                 sample_rate_list: List,
                 clip_seg_num_probability_list: List = None,
                 sample_rate_probability_list: List = None) -> None:
        super().__init__()
        if clip_seg_num_probability_list is not None:
            assert len(clip_seg_num_list) == len(clip_seg_num_probability_list), f"length of clip_seg_num_list and length of clip_seg_num_probability_list must be same!"
        if sample_rate_probability_list is not None:
            assert len(sample_rate_list) == len(sample_rate_probability_list), f"length of sample_rate_list and length of sample_rate_probability_list must be same!"
        self.clip_seg_num_list = clip_seg_num_list
        self.clip_seg_num_probability_list = clip_seg_num_probability_list
        self.sample_rate_list = sample_rate_list
        self.sample_rate_probability_list = sample_rate_probability_list
    
    def __next__(self) -> Dict:
        if self.cur_len < self.max_len:
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

            before_len = self.cur_len
            if self.cur_len + sample_rate * clip_seg_num < self.max_len:
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
            else:
                left_len = self.max_len - self.cur_len
                clip_seg_num = math.ceil(left_len / sample_rate)
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
        raise StopIteration

@AbstractBuildFactory.register('dynamic_stream_generator')
class ListLoopChoiceDynamicStreamGenerator(BaseDynamicStreamGenerator):
    def __init__(self,
                 clip_seg_num_list: List,
                 sample_rate_list: List,
                 clip_seg_num_probability_list: List = None,
                 sample_rate_probability_list: List = None) -> None:
        super().__init__()
        if clip_seg_num_probability_list is not None:
            assert len(clip_seg_num_list) == len(clip_seg_num_probability_list), f"length of clip_seg_num_list and length of clip_seg_num_probability_list must be same!"
        if sample_rate_probability_list is not None:
            assert len(sample_rate_list) == len(sample_rate_probability_list), f"length of sample_rate_list and length of sample_rate_probability_list must be same!"
        self.clip_seg_num_list = clip_seg_num_list
        self.clip_seg_num_probability_list = clip_seg_num_probability_list
        self.sample_rate_list = sample_rate_list
        self.sample_rate_probability_list = sample_rate_probability_list
        self.clip_seg_num_idx = 0
        self.sample_rate_idx = 0
    
    def __next__(self) -> Dict:
        if self.cur_len < self.max_len:
            clip_seg_num = self.clip_seg_num_list[self.clip_seg_num_idx]
            sample_rate = self.sample_rate_list[self.sample_rate_idx]
            self.clip_seg_num_idx += 1
            self.sample_rate_idx += 1

            if self.clip_seg_num_idx >= len(self.clip_seg_num_list):
                self.clip_seg_num_idx = 0
            if self.sample_rate_idx >= len(self.sample_rate_list):
                self.sample_rate_idx = 0

            before_len = self.cur_len
            if self.cur_len + sample_rate * clip_seg_num < self.max_len:
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
            else:
                left_len = self.max_len - self.cur_len
                clip_seg_num = math.ceil(left_len / sample_rate)
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
        raise StopIteration

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
        if self.cur_len < self.max_len:
            clip_seg_num = random.randint(self.clip_seg_num_min, self.clip_seg_num_max)
            sample_rate = random.randint(self.sample_rate_min, self.sample_rate_max)

            before_len = self.cur_len
            if self.cur_len + sample_rate * clip_seg_num < self.max_len:
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
            else:
                left_len = self.max_len - self.cur_len
                clip_seg_num = math.ceil(left_len / sample_rate)
                self.cur_len += sample_rate * clip_seg_num
                self.update_precise_sliding_num(self.cur_len)
                return dict(sample_rate=sample_rate, clip_seg_num=clip_seg_num, currenct_frame_idx=before_len)
        raise StopIteration
            
class DynamicStreamDataset(StreamDataset):
    dynamic_stream_generator: BaseDynamicStreamGenerator
    def __init__(self,
                 dynamic_stream_generator: Dict,
                 **kwargs) -> None:
        self.dynamic_stream_generator = AbstractBuildFactory.create_factory('dynamic_stream_generator').create(dynamic_stream_generator)
        super().__init__(**kwargs)
    
    def _step_sliding_sampler(self, woker_id, num_workers, info_list):
        # dispatch function
        current_sliding_cnt = woker_id * self.temporal_clip_batch_size
        mini_sliding_cnt = 0
        next_step_flag = False
        for step, sliding_num, dynamic_sample_list, info in info_list:
            while current_sliding_cnt < sliding_num and len(info) > 0:
                while mini_sliding_cnt < self.temporal_clip_batch_size:
                    if current_sliding_cnt < sliding_num:
                        for single_info in info:
                            single_info.update(dynamic_sample_list[current_sliding_cnt])
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