'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:30:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-18 19:56:36
Description  : feature sampler
FilePath     : /ETESVS/loader/sampler/feature_sampler.py
'''
import numpy as np
import random
from ..builder import SAMPLER


class FeatureFrameSample():
    def __init__(self, mode='random'):
        assert mode in ['random', 'uniform'], 'not support mode'
        self.mode = mode
    
    def random_sample(self, start_idx, end_idx, sample_rate):
        sample_idx = list(
                random.sample(list(range(start_idx, end_idx)),
                    len(list(range(start_idx, end_idx, sample_rate)))))
        sample_idx.sort()
        return sample_idx

    def uniform_sample(self, start_idx, end_idx, sample_rate):
        return list(range(start_idx, end_idx, sample_rate))
        
    def __call__(self, start_idx, end_idx, sample_rate):
        if self.mode == 'random':
            return self.random_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'uniform':
            return self.uniform_sample(start_idx, end_idx, sample_rate)
        else:
            raise NotImplementedError

@SAMPLER.register()
class FeatureStreamSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 sample_rate=1,
                 clip_seg_num=15,
                 sliding_window=60,
                 ignore_index=-100,
                 sample_mode='random',
                 format="NTC"):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.format = format
        self.sample = FeatureFrameSample(mode = sample_mode)
    
    def _all_valid_frames(self, start_frame, end_frame, feature_len, feature, labels):
        vid_end_frame = end_frame
        if end_frame > feature_len:
            vid_end_frame = feature_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()
        frames_feature = feature[:, frames_idx]
        mask = np.ones((labels.shape[0]))

        return frames_feature, labels, mask
    
    def _some_valid_frames(self, start_frame, end_frame, feature_len, frames_len, feature, labels):
        frames_idx = self.sample(start_frame, feature_len, self.sample_rate)
        label_frames_idx = self.sample(start_frame, frames_len, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=frames_len, samples_idx=label_frames_idx).copy()
        frames_feature = feature[:, frames_idx]
        pad_len = self.clip_seg_num - frames_feature.shape[-1]
        frames_feature = np.concatenate([frames_feature, np.zeros((2048, pad_len))], axis=-1)
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])
        
        return frames_feature, labels, mask

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
        frames_len = int(results['frames_len'])
        feature_len = int(results['feature_len'])
        results['frames_len'] = frames_len
        feature = results['frames']
        labels = results['raw_labels']

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len and end_frame < frames_len:
            frames_feature, labels, mask = self._all_valid_frames(start_frame, end_frame, feature_len, feature, labels)
        elif start_frame < frames_len and end_frame >= frames_len:
            frames_feature, labels, mask = self._some_valid_frames(start_frame, end_frame, feature_len, frames_len, feature, labels)
        else:
            frames_feature = np.zeros((2048, self.clip_seg_num))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)
        
        if self.format in ["NTC"]:
            frames_feature = frames_feature[:, :self.clip_seg_num].T
        else:
            frames_feature = frames_feature[:, :self.clip_seg_num]

        results['feature'] = frames_feature.copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        return results

@SAMPLER.register()
class FeatureSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 sample_rate=1,
                 ignore_index=-100,
                 sample_mode='random',
                 format="NTC"
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.ignore_index = ignore_index
        self.format = format
        self.sample = FeatureFrameSample(mode = sample_mode)

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
        frames_len = int(results['frames_len'])
        feature_len = int(results['feature_len'])
        results['frames_len'] = frames_len
        feature = results['frames']
        labels = results['raw_labels']

        # generate sample index
        if frames_len < feature_len:
            frames_idx = self.sample(0, frames_len, self.sample_rate)
            labels = self._labels_sample(labels, start_frame=0, end_frame=frames_len, samples_idx=frames_idx).copy()
            frames_feature = feature[:, frames_idx]
            mask = np.ones((labels.shape[0]), dtype=np.float32)
        else:
            frames_idx = self.sample(0, feature_len, self.sample_rate)
            labels = self._labels_sample(labels, start_frame=0, end_frame=feature_len, samples_idx=frames_idx).copy()
            frames_feature = feature[:, frames_idx]
            mask = np.ones((labels.shape[0]), dtype=np.float32)

        if self.format in ["NTC"]:
            frames_feature = frames_feature.T

        results['feature'] = frames_feature.copy()
        results['labels'] = labels.copy()
        results['masks'] = mask.copy()
        return results