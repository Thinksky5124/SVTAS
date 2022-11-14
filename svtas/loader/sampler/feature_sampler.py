'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:30:34
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 15:01:22
Description  : feature sampler
FilePath     : /SVTAS/svtas/loader/sampler/feature_sampler.py
'''
import numpy as np
from .frame_sampler import FrameIndexSample

from ..builder import SAMPLER

@SAMPLER.register()
class FeatureStreamSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 feature_dim_dict={"feature":2048},
                 is_train=False,
                 sample_rate_dict={"feature":4, "labels":4},
                 clip_seg_num_dict={"feature":15, "labels":15},
                 sliding_window_dict={"feature":60, "labels":60},
                 sample_add_key_pair={"frames":"feature"},
                 ignore_index=-100,
                 sample_mode='random',
                 format="NTC",
                 frame_idx_key='sample_sliding_idx'):
        assert len(sample_rate_dict)==len(clip_seg_num_dict)==len(sliding_window_dict)==(len(sample_add_key_pair)+1)
        
        self.sample_rate_dict = sample_rate_dict
        self.is_train = is_train
        self.clip_seg_num_dict = clip_seg_num_dict
        self.sliding_window_dict = sliding_window_dict
        self.sample_add_key_pair = sample_add_key_pair
        self.ignore_index = ignore_index
        self.frame_idx_key = frame_idx_key
        assert format in ['NTC', 'NCT', 'NCTHW']
        self.format = format
        self.feature_dim_dict = feature_dim_dict
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
        results['masks'] = mask.astype(np.float32).copy()
        return results
    
    def _get_start_end_frame_idx(self, results, sample_rate, sample_num, sliding_windows):
        frames_len = int(results['frames_len'])
        feature_len = int(results['feature_len'])
        small_frames_video_len = min(frames_len, feature_len)

        # generate sample index
        if self.frame_idx_key in results.keys():
            start_frame = results[self.frame_idx_key] * sliding_windows
            end_frame = start_frame + sample_num * sample_rate
        else:
            start_frame = 0
            end_frame = small_frames_video_len
        
        small_end_frame_idx = min(end_frame, small_frames_video_len)
        return start_frame, small_end_frame_idx
    
    def _sample_frames(self, results, sample_rate, feature_dim, sample_num, sliding_windows, add_key='feature', sample_key='frames'):
        container = results[sample_key]
        filename = results['filename']
        start_frame, end_frame = self._get_start_end_frame_idx(results, sample_rate, sample_num, sliding_windows)

        if start_frame < end_frame:
            frames_idx = self.sample(start_frame, end_frame, sample_rate, sample_num)
            try:
                frames_feature = container.get_batch(frames_idx)
            except:
                print("file: " + filename + " sample frame index: " + ",".join([str(i) for i in frames_idx]) +" error!")
                raise
            pad_len = sample_num - frames_feature.shape[-1]
            if pad_len > 0:
                frames_feature = np.concatenate([frames_feature, np.zeros((feature_dim, pad_len))], axis=-1)
        else:
            frames_feature = np.zeros((feature_dim, sample_num))

        if self.format in ["NTC"]:
            frames_feature = frames_feature.T

        results[add_key] = frames_feature.copy()
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
            frames_len: length of frames.
        return:
            sampling id.
        """
        for sample_key, add_key in self.sample_add_key_pair.items():
            feature_dim = self.feature_dim_dict[add_key]
            sample_rate = self.sample_rate_dict[add_key]
            clip_seg_num = self.clip_seg_num_dict[add_key]
            sliding_window = self.sliding_window_dict[add_key]
            results = self._sample_frames(results, sample_rate, feature_dim, clip_seg_num, sliding_window, add_key=add_key, sample_key=sample_key)
        sample_rate = self.sample_rate_dict["labels"]
        clip_seg_num = self.clip_seg_num_dict["labels"]
        sliding_window = self.sliding_window_dict["labels"]
        results = self._sample_label(results, sample_rate, clip_seg_num, sliding_window, add_key='labels', sample_key='raw_labels')

        return results

@SAMPLER.register()
class FeatureSampler(FeatureStreamSampler):
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """
    def __init__(self,
                 feature_dim_dict={ "feature": 2048},
                 is_train=False,
                 sample_rate_dict={ "feature": 1,"labels": 1 },
                 sample_add_key_pair={ "frames": "feature" },
                 ignore_index=-100,
                 sample_mode='random',
                 format="NTC"):
        super().__init__(feature_dim_dict=feature_dim_dict,
                         is_train=is_train,
                         sample_rate_dict=sample_rate_dict,
                         clip_seg_num_dict={ "feature": -1,"labels": -1 },
                         sliding_window_dict={ "feature": 1000,"labels": 1000 },
                         sample_add_key_pair=sample_add_key_pair,
                         ignore_index=ignore_index,
                         sample_mode=sample_mode,
                         format=format)
