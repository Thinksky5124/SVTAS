'''
Author       : Thyssen Wen
Date         : 2022-05-28 10:58:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-28 11:28:01
Description  : Transeger Sampler
FilePath     : /ETESVS/loader/sampler/transeger_sampler.py
'''
import numpy as np
from PIL import Image
from ..builder import SAMPLER
from .frame_sampler import VideoFrameSample

@SAMPLER.register()
class VideoStreamTransegerSampler():
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
                 sample_mode='random'
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.sample = VideoFrameSample(mode = sample_mode)
    
    def _all_valid_frames(self, start_frame, end_frame, video_len, frames_len, container, labels):
        imgs = []
        vid_end_frame = end_frame
        if end_frame > video_len:
            vid_end_frame = video_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels, last_clip_labels = self._transeger_labels_sample(labels, start_frame=start_frame, end_frame=end_frame, labels_len=frames_len, samples_idx=frames_idx).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        if len(imgs) < self.clip_seg_num:
            np_frames = np_frames[-1].asnumpy().copy()
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
                
        mask = np.ones((labels.shape[0]))

        return imgs, labels, last_clip_labels, mask
    
    def _some_valid_frames(self, start_frame, end_frame, video_len, frames_len, container, labels):
        imgs = []
        frames_idx = self.sample(start_frame, video_len, self.sample_rate)
        label_frames_idx = self.sample(start_frame, frames_len, self.sample_rate)
        labels, last_clip_labels = self._transeger_labels_sample(labels, start_frame=start_frame, end_frame=frames_len, labels_len=frames_len, samples_idx=label_frames_idx).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))
        np_frames = np.zeros_like(np_frames[0])
        pad_len = self.clip_seg_num - len(imgs)
        for i in range(pad_len):
            imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])

        return imgs, labels, last_clip_labels, mask
    
    def _transeger_labels_sample(self, labels, start_frame=0, end_frame=0, labels_len=0, samples_idx=[]):
        if self.is_train:
            sample_labels = labels[samples_idx]
            sample_labels = np.repeat(sample_labels, repeats=self.sample_rate, axis=-1)

            last_clip_labels_start = samples_idx[0] - self.clip_seg_num * self.sample_rate
            if last_clip_labels_start < 0:
                last_clip_labels_start = 0
            last_clip_labels = labels[last_clip_labels_start:samples_idx[0]]
        else:
            sample_labels = labels[start_frame:end_frame]

            last_clip_labels_start = start_frame - self.clip_seg_num * self.sample_rate
            if last_clip_labels_start < 0:
                last_clip_labels_start = 0
            last_clip_labels = labels[last_clip_labels_start:start_frame]
        
        if last_clip_labels.shape[0] < self.clip_seg_num * self.sample_rate:
            pad_len = self.clip_seg_num * self.sample_rate - last_clip_labels.shape[0]
            last_clip_labels = np.concatenate([np.full((pad_len), self.ignore_index), last_clip_labels])

        return sample_labels, last_clip_labels

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        video_len = int(results['video_len'])
        results['frames_len'] = frames_len
        container = results['frames']
        labels = results['raw_labels']

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len and end_frame < frames_len:
            imgs, labels, last_clip_labels, mask = self._all_valid_frames(start_frame, end_frame, video_len, frames_len, container, labels)
        elif start_frame < frames_len and end_frame >= frames_len:
            imgs, labels, last_clip_labels, mask = self._some_valid_frames(start_frame, end_frame, video_len, frames_len, container, labels)
        else:
            imgs = []
            np_frames = np.zeros((240, 320, 3))
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)
            last_clip_labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['last_clip_labels'] = last_clip_labels.copy()
        results['mask'] = mask.copy()
        return results