'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:32:33
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-01 10:40:35
Description  : Raw frame sampler
FilePath     : /SVTAS/svtas/loader/sampler/frame_sampler.py
'''
import copy
import random

import numpy as np
from PIL import Image

from ..builder import SAMPLER


class FrameIndexSample():
    def __init__(self, mode='random'):
        assert mode in ['random', 'uniform', 'linspace', 'random_choice', 'uniform_random'], 'not support mode'
        self.mode = mode
    
    def random_sample(self, start_idx, end_idx, sample_rate):
        sample_idx = list(
                random.sample(list(range(start_idx, end_idx)),
                    len(list(range(start_idx, end_idx, sample_rate)))))
        sample_idx.sort()
        return sample_idx

    def uniform_sample(self, start_idx, end_idx, sample_rate):
        return list(range(start_idx, end_idx, sample_rate))
    
    def uniform_random_sample(self, start_idx, end_idx, sample_num, sample_rate):
        sample_index = np.ceil(np.linspace(start_idx, end_idx, num=sample_num)).astype(np.int64) + np.random.randint(0, high=sample_rate, size=sample_num)
        sample_index[sample_index >= end_idx] = end_idx - 1
        return list(sample_index)

    def linspace_sample(self, start_idx, end_idx, sample_num):
        return list(np.ceil(np.linspace(start_idx, end_idx, num=sample_num)).astype(np.int64))
    
    def random_choice_sample(self, start_idx, end_idx, sample_num):
        return list(random.sample(range(start_idx, end_idx), sample_num)).sort()

    def __call__(self, start_idx, end_idx, sample_rate, sample_num):
        if self.mode == 'random':
            return self.random_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'uniform':
            return self.uniform_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'linspace':
            return self.linspace_sample(start_idx, end_idx - 1, sample_num)
        elif self.mode == 'random_choice':
            return self.random_choice_sample(start_idx, end_idx, sample_num)
        elif self.mode == 'uniform_random':
            return self.uniform_random_sample(start_idx, end_idx, sample_num, sample_rate)
        else:
            raise NotImplementedError

@SAMPLER.register()
class VideoStreamSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 sample_rate_dict={"imgs":4, "labels":4},
                 clip_seg_num_dict={"imgs":15, "labels":15},
                 sliding_window_dict={"imgs":60, "labels":60},
                 ignore_index=-100,
                 sample_add_key_pair={"frames":"imgs"},
                 channel_mode_dict={"imgs":"RGB", "res":"RGB", "flows":"XY"},
                 channel_num_dict={"imgs":3, "res":3, "flows":2},
                 sample_mode='random',
                 frame_idx_key='sample_sliding_idx'
                 ):
        # assert len(sample_rate_dict)==len(clip_seg_num_dict)==len(sliding_window_dict)==(len(sample_add_key_pair)+1)

        self.sample_rate_dict = sample_rate_dict
        self.is_train = is_train
        self.clip_seg_num_dict = clip_seg_num_dict
        self.sliding_window_dict = sliding_window_dict
        self.ignore_index = ignore_index
        self.channel_mode_dict = channel_mode_dict
        self.frame_idx_key = frame_idx_key
        self.sample_add_key_pair = sample_add_key_pair
        self.channel_num_dict = channel_num_dict
        self.sample = FrameIndexSample(mode = sample_mode)
    
    def _sample_label(self, results, sample_rate, sample_num, sliding_windows, add_key='labels', sample_key='raw_labels'):
        container = results[sample_key]
        start_frame, end_frame = self._get_start_end_frame_idx(results, sample_rate, sample_num, sliding_windows)
        
        if start_frame < end_frame:
            frames_idx = self.sample(start_frame, end_frame, sample_rate, sample_num)
            labels = self._labels_sample(container, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx, sample_rate=sample_rate).copy()

            vaild_mask = np.ones((labels.shape[0]))
            mask_pad_len = sample_num * sample_rate - labels.shape[0]
            if mask_pad_len > 0:
                void_mask = np.zeros((mask_pad_len))
                mask = np.concatenate([vaild_mask, void_mask], axis=0)
                labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])
            else:
                mask = vaild_mask
        else:
            labels = np.full((sample_num * sample_rate), self.ignore_index)
            mask = np.zeros((sample_num * sample_rate))

        results[add_key] = labels.copy()
        results['masks'] = mask.copy()
        return results
    
    def _get_start_end_frame_idx(self, results, sample_rate, sample_num, sliding_windows):
        frames_len = int(results['frames_len'])
        video_len = int(results['video_len'])
        small_frames_video_len = min(frames_len, video_len)

        # generate sample index
        if self.frame_idx_key in results.keys():
            start_frame = results[self.frame_idx_key] * sliding_windows
            end_frame = start_frame + sample_num * sample_rate
        else:
            start_frame = 0
            end_frame = small_frames_video_len
        
        small_end_frame_idx = min(end_frame, small_frames_video_len)
        return start_frame, small_end_frame_idx

    def _sample_2_PIL_frame(self, frames_select, channel_mode, channel_num, sample_num):
        imgs = []
        np_frames = frames_select.asnumpy()
        if np_frames.shape[0] > 0:
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i].copy()
                imgs.append(Image.fromarray(imgbuf, mode=channel_mode))
            np_frames = np.zeros_like(np_frames[0])
        else:
            np_frames = np.zeros((224, 224, channel_num))
        pad_len = sample_num - len(imgs)
        if pad_len > 0:
            for i in range(max(0, pad_len)):
                imgs.append(Image.fromarray(np_frames, mode=channel_mode))
        return imgs
    
    def _sample_2_numpy_frame(self, frames_select, channel_mode, channel_num, sample_num):
        imgs = []
        np_frames = frames_select
        if np_frames.shape[0] > 0:
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i].copy()
                imgs.append(imgbuf)
            np_frames = np.zeros_like(np_frames[0])
        else:
            np_frames = np.zeros((224, 224, channel_num))
        pad_len = sample_num - len(imgs)
        if pad_len > 0:
            for i in range(max(0, pad_len)):
                imgs.append(np_frames)
        return imgs
    
    def _sample_2_numpy_frame_for_none(self, channel_mode, channel_num, sample_num):
        imgs = []
        np_frames = np.zeros((224, 224, channel_num))
        for i in range(sample_num):
            imgs.append(np_frames)
        return imgs
    
    def _sample_2_PIL_frame_for_none(self, channel_mode, channel_num, sample_num):
        imgs = []
        np_frames = np.zeros((224, 224, channel_num))
        for i in range(sample_num):
            imgs.append(Image.fromarray(np_frames, mode=channel_mode))
        return imgs
    
    def _sample_dict_2_frame(self, frames_select, results, sample_num):
        for k, v in frames_select.items():
            frames = self._sample_2_numpy_frame(frames_select=v, channel_mode=self.channel_mode_dict[k], channel_num=self.channel_num_dict[k], sample_num=sample_num)
            results[k] = frames.copy()
        return results
    
    def _sample_dict_2_frame_for_none(self, container, results, sample_num):
        for key in container.dict_keys:
            frames = self._sample_2_numpy_frame_for_none(channel_mode=self.channel_mode_dict[key], channel_num=self.channel_num_dict[key], sample_num=sample_num)
            results[key] = frames.copy()
        return results
    
    def _sample_frames(self, results, sample_rate, channel_mode, channel_num, sample_num, sliding_windows, add_key='imgs', sample_key='frames'):
        container = results[sample_key]
        filename = results['filename']
        start_frame, end_frame = self._get_start_end_frame_idx(results, sample_rate, sample_num, sliding_windows)

        if start_frame < end_frame:
            frames_idx = self.sample(start_frame, end_frame, sample_rate, sample_num)
            try:
                frames_select = container.get_batch(frames_idx)
            except:
                print("file: " + filename + " sample frame index: " + ",".join([str(i) for i in frames_idx]) +" error!")
                raise
            # dearray_to_img
            if container.out_dtype == "PIL":
                imgs = self._sample_2_PIL_frame(frames_select=frames_select, channel_mode=channel_mode, channel_num=channel_num, sample_num=sample_num)
            elif container.out_dtype == "numpy":
                imgs = self._sample_2_numpy_frame(frames_select=frames_select, channel_mode=channel_mode, channel_num=channel_num, sample_num=sample_num)
            elif container.out_dtype == "dict":
                return self._sample_dict_2_frame(frames_select=frames_select, results=results, sample_num=sample_num)
            else:
                raise NotImplementedError
        else:
            if container.out_dtype == "numpy":
                imgs = self._sample_2_numpy_frame_for_none(channel_mode=channel_mode, channel_num=channel_num, sample_num=sample_num)
            elif container.out_dtype == "PIL":
                imgs = self._sample_2_PIL_frame_for_none(channel_mode=channel_mode, channel_num=channel_num, sample_num=sample_num)
            elif container.out_dtype == "dict":
                return self._sample_dict_2_frame_for_none(container=container, results=results, sample_num=sample_num)
            else:
                raise NotImplementedError

        results[add_key] = imgs[:sample_num].copy()
        return results

    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[], sample_rate=1):
        if self.is_train:
            sample_labels = labels[samples_idx]
            sample_labels = np.repeat(sample_labels, repeats=sample_rate, axis=-1)
        else:
            sample_labels = labels[start_frame:end_frame]
        return sample_labels
    
    def __call__(self, results):
        """
        Args:
            results: data dict.
        return:
           data dict.
        """
        for sample_key, add_key in self.sample_add_key_pair.items():
            channel_mode = self.channel_mode_dict[add_key]
            channel = self.channel_num_dict[add_key]
            sample_rate = self.sample_rate_dict[add_key]
            clip_seg_num = self.clip_seg_num_dict[add_key]
            sliding_window = self.sliding_window_dict[add_key]
            results = self._sample_frames(results, sample_rate, channel_mode, channel, clip_seg_num, sliding_window, add_key=add_key, sample_key=sample_key)
        sample_rate = self.sample_rate_dict["labels"]
        clip_seg_num = self.clip_seg_num_dict["labels"]
        sliding_window = self.sliding_window_dict["labels"]
        results = self._sample_label(results, sample_rate, clip_seg_num, sliding_window, add_key='labels', sample_key='raw_labels')

        return results

@SAMPLER.register()
class VideoSampler(VideoStreamSampler):
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 clip_seg_num_dict={"imgs":15, "labels":15},
                 ignore_index=-100,
                 sample_add_key_pair={"frames":"imgs"},
                 channel_mode_dict={"imgs":"RGB"},
                 sample_mode='linspace',
                 frame_idx_key='sample_sliding_idx'):
        super().__init__(is_train=is_train,
                         sample_rate_dict={"imgs":1, "labels":1},
                         clip_seg_num_dict=clip_seg_num_dict,
                         sample_add_key_pair=sample_add_key_pair,
                         sliding_window_dict={"imgs":1000, "labels":1000},
                         ignore_index=ignore_index,
                         channel_mode_dict=channel_mode_dict,
                         sample_mode=sample_mode,
                         frame_idx_key=frame_idx_key)

@SAMPLER.register()
class VideoClipSampler(VideoStreamSampler):
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 clip_seg_num_dict={"imgs":15, "labels":15},
                 sample_rate_dict={"imgs":1, "labels":1},
                 ignore_index=-100,
                 random_temporal_agument=False,
                 sample_add_key_pair={"frames":"imgs"},
                 channel_mode_dict={"imgs":"RGB"},
                 sample_mode='linspace',
                 frame_idx_key='sample_sliding_idx'):
        super().__init__(is_train=is_train,
                         sample_rate_dict=sample_rate_dict,
                         clip_seg_num_dict=clip_seg_num_dict,
                         sample_add_key_pair=sample_add_key_pair,
                         sliding_window_dict={"imgs":1000, "labels":1000},
                         ignore_index=ignore_index,
                         channel_mode_dict=channel_mode_dict,
                         sample_mode=sample_mode,
                         frame_idx_key=frame_idx_key)
        self.random_temporal_agument = random_temporal_agument
    
    def _sample_start_end_idx(self, results):
        frames_len = int(results['frames_len'])
        video_len = int(results['video_len'])
        small_frames_video_len = min(frames_len, video_len)

        start_frame = int(results['start_frame'])
        end_frame = int(results['end_frame'])

        if self.random_temporal_agument:
            clip_seg_num = self.clip_seg_num_dict[list(self.sample_add_key_pair.values())[0]]
            random_temporal_shift = random.randint(-clip_seg_num, clip_seg_num)
            if start_frame + random_temporal_shift >= 0:
                start_frame = start_frame + random_temporal_shift
                end_frame = end_frame + random_temporal_shift

        if end_frame > small_frames_video_len:
            start_frame = small_frames_video_len - (end_frame - start_frame)
            end_frame = small_frames_video_len

        results['start_frame'] = start_frame
        results['end_frame'] = end_frame
        return results

    def _get_start_end_frame_idx(self, results, sample_rate, sample_num, sliding_windows):

        start_frame = int(results['start_frame'])
        end_frame = int(results['end_frame'])

        return start_frame, end_frame
    
    def __call__(self, results):
        """
        Args:
            results: data dict.
        return:
           data dict.
        """
        results = self._sample_start_end_idx(results)
        for sample_key, add_key in self.sample_add_key_pair.items():
            channel_mode = self.channel_mode_dict[add_key]
            channel = self.channel_num_dict[add_key]
            sample_rate = self.sample_rate_dict[add_key]
            clip_seg_num = self.clip_seg_num_dict[add_key]
            sliding_window = self.sliding_window_dict[add_key]
            results = self._sample_frames(results, sample_rate, channel_mode, channel, clip_seg_num, sliding_window, add_key=add_key, sample_key=sample_key)
        sample_rate = self.sample_rate_dict["labels"]
        clip_seg_num = self.clip_seg_num_dict["labels"]
        sliding_window = self.sliding_window_dict["labels"]
        results = self._sample_label(results, sample_rate, clip_seg_num, sliding_window, add_key='labels', sample_key='raw_labels')

        return results