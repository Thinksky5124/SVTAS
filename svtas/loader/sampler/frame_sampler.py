'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:32:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 13:47:01
Description  : Raw frame sampler
FilePath     : /SVTAS/svtas/loader/sampler/frame_sampler.py
'''
import random

import numpy as np
from PIL import Image

from ..builder import SAMPLER


class FrameSample():
    def __init__(self, mode='random'):
        assert mode in ['random', 'uniform', 'linspace', 'random_choice'], 'not support mode'
        self.mode = mode
    
    def random_sample(self, start_idx, end_idx, sample_rate):
        sample_idx = list(
                random.sample(list(range(start_idx, end_idx)),
                    len(list(range(start_idx, end_idx, sample_rate)))))
        sample_idx.sort()
        return sample_idx

    def uniform_sample(self, start_idx, end_idx, sample_rate):
        return list(range(start_idx, end_idx, sample_rate))
    
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
                 sample_rate=4,
                 clip_seg_num=15,
                 sliding_window=60,
                 ignore_index=-100,
                 channel_mode="RGB",
                 sample_mode='random',
                 frame_idx_key='sample_sliding_idx'
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.frame_idx_key = frame_idx_key
        if self.channel_mode == "RGB":
            self.channel = 3
        elif self.channel_mode == "XY":
            self.channel = 2
        self.sample = FrameSample(mode = sample_mode)
    
    def _sample_label(self, results, sample_rate, sample_num, sliding_windows, add_key='labels', sample_key='raw_labels'):
        container = results[sample_key]
        start_frame, end_frame = self._get_start_end_frame_idx(results, sample_rate, sample_num, sliding_windows)
        
        if start_frame < end_frame:
            frames_idx = self.sample(start_frame, end_frame, sample_rate, sample_num)
            labels = self._labels_sample(container, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()

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

    
    def _sample_frames(self, results, sample_rate, channel_mode, channel_num, sample_num, sliding_windows, add_key='imgs', sample_key='frames'):
        imgs = []
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
            elif container.out_dtype == "numpy":
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
            elif container.out_dtype == "dict":
                pass
            else:
                raise NotImplementedError
        else:
            np_frames = np.zeros((224, 224, channel_num))
            if container.out_dtype == "numpy":
                for i in range(sample_num):
                    imgs.append(np_frames)
            elif container.out_dtype == "PIL":
                for i in range(sample_num):
                    imgs.append(Image.fromarray(np_frames, mode=channel_mode))
            elif container.out_dtype == "dict":
                pass
            else:
                raise NotImplementedError

        results[add_key] = imgs[:sample_num].copy()
        return results

    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[]):
        if self.is_train:
            sample_labels = labels[samples_idx]
            sample_labels = np.repeat(sample_labels, repeats=self.sample_rate, axis=-1)
        else:
            sample_labels = labels[start_frame:end_frame]
        return sample_labels
    
    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        results = self._sample_frames(results, self.sample_rate, self.channel_mode, self.channel, self.clip_seg_num, self.sliding_window, add_key='imgs', sample_key='frames')
        results = self._sample_label(results, self.sample_rate, self.clip_seg_num, self.sliding_window, add_key='labels', sample_key='raw_labels')

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
                 clip_seg_num=15,
                 ignore_index=-100,
                 channel_mode="RGB",
                 sample_mode='linspace',
                 frame_idx_key='sample_sliding_idx'):
        super().__init__(is_train=is_train,
                         sample_rate=1,
                         clip_seg_num=clip_seg_num,
                         sliding_window=1000,
                         ignore_index=ignore_index,
                         channel_mode=channel_mode,
                         sample_mode=sample_mode,
                         frame_idx_key=frame_idx_key)

@SAMPLER.register()
class RGBFlowVideoStreamSampler(VideoStreamSampler):
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 is_train=False,
                 sample_rate=4,
                 clip_seg_num=15,
                 sliding_window=60,
                 ignore_index=-100,
                 rgb_channel_mode="RGB",
                 flow_channel_mode="XY",
                 sample_mode='random',
                 frame_idx_key='sample_sliding_idx'):
        super().__init__(is_train,
                         sample_rate,
                         clip_seg_num,
                         sliding_window, 
                        ignore_index,
                        rgb_channel_mode,
                        sample_mode,
                        frame_idx_key=frame_idx_key)
        self.flow_channel_mode = flow_channel_mode
        if self.flow_channel_mode == "XY":
            self.flow_channel = 2
        elif self.flow_channel_mode == "RGB":
            self.flow_channel = 3


    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """

        results = self._sample_frames(results, self.sample_rate, self.channel_mode, self.channel, self.clip_seg_num, self.sliding_window, add_key='imgs', sample_key='rgb_frames')
        results = self._sample_frames(results, self.sample_rate, self.flow_channel_mode, self.flow_channel, self.clip_seg_num, self.sliding_window, add_key='flows', sample_key='flow_frames')
        results = self._sample_label(results, self.sample_rate, self.clip_seg_num, self.sliding_window, add_key='labels', sample_key='raw_labels')

        return results

@SAMPLER.register()
class RGBMVIPBVideoStreamSampler(VideoStreamSampler):
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 is_train=False,
                 gop_size=15,
                 sample_rate=4,
                 rgb_clip_seg_num=1,
                 flow_clip_seg_num=15,
                 rgb_sliding_window=60,
                 flow_sliding_window=60,
                 ignore_index=-100,
                 rgb_channel_mode="RGB",
                 flow_channel_mode="XY",
                 sample_mode='random',
                 frame_idx_key='sample_sliding_idx'):
        super().__init__(is_train,
                         sample_rate,
                         flow_clip_seg_num,
                         flow_sliding_window, 
                         ignore_index,
                         flow_channel_mode,
                         sample_mode,
                         frame_idx_key=frame_idx_key)
        assert gop_size < self.clip_seg_num, "GOP size: " + str(gop_size) + " must samller than clip_seg_num: " + str(self.clip_seg_num) + " !"
        self.gop_size = gop_size
        self.rgb_clip_seg_num = rgb_clip_seg_num
        self.rgb_channel_mode = rgb_channel_mode
        self.rgb_sliding_window = rgb_sliding_window
        self.flow_channel_mode = flow_channel_mode
        if self.flow_channel_mode == "XY":
            self.flow_channel = 2
        elif self.flow_channel_mode == "RGB":
            self.flow_channel = 3

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """

        results = self._sample_frames(results, self.sample_rate * self.gop_size, self.rgb_channel_mode, self.channel,self.rgb_clip_seg_num, self.rgb_sliding_window, add_key='imgs', sample_key='rgb_frames')
        results = self._sample_frames(results, self.sample_rate, self.channel_mode, self.flow_channel,self.clip_seg_num, self.sliding_window, add_key='flows', sample_key='flow_frames')
        results = self._sample_label(results, self.sample_rate, self.clip_seg_num, self.sliding_window, add_key='labels', sample_key='raw_labels')

        return results

