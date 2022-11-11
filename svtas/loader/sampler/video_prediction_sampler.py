'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:58:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 14:52:21
Description  : Video Prediction Sampler
FilePath     : /SVTAS/svtas/loader/sampler/video_prediction_sampler.py
'''
import numpy as np
from PIL import Image

from ..builder import SAMPLER
from .feature_sampler import FeatureStreamSampler
from .frame_sampler import VideoStreamSampler


@SAMPLER.register()
class VideoPredictionFeatureStreamSampler(FeatureStreamSampler):
    def __init__(self,
                 pred_clip_seg_num=8,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.pred_clip_seg_num = pred_clip_seg_num
    
    def _get_start_end_pred_frame_idx(self, results, sample_rate, pred_sample_num, sample_num, sliding_windows):
        start_frame = results[self.frame_idx_key] * sliding_windows + sample_num * sample_rate
        end_frame = start_frame + pred_sample_num * sample_rate
        return start_frame, end_frame
    
    def _sample_pred_label(self, results, sample_rate, pred_sample_num, sample_num, sliding_windows, add_key='pred_labels', sample_key='raw_labels'):
        container = results[sample_key]
        start_frame, end_frame = self._get_start_end_pred_frame_idx(results, sample_rate, pred_sample_num, sample_num, sliding_windows)
        frames_len = int(results['frames_len'])

        if start_frame < frames_len:
            pred_label_frames_idx = self.sample(start_frame, frames_len, sample_rate)
            pred_labels = self._labels_sample(container, start_frame=start_frame, end_frame=end_frame, samples_idx=pred_label_frames_idx, sample_rate=sample_rate).copy()
            pad_len = pred_sample_num * sample_rate - pred_labels.shape[0]
            if pad_len > 0:
                pred_labels = np.concatenate([pred_labels, np.full((pad_len), self.ignore_index)])
        else:
            pred_labels = np.full((pred_sample_num * sample_rate), self.ignore_index)

        results[add_key] = pred_labels.copy()
        return results
    
    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        for sample_key, add_key in self.sample_add_key_pair.items():
            channel_mode = self.channel_mode_dict[add_key]
            channel = self.channel_num_dict[add_key]
            sample_rate = self.sample_rate_dict[add_key]
            clip_seg_num = self.clip_seg_num_dict[add_key]
            sliding_window = self.sliding_window_dict[add_key]
            results = self._sample_frames(results, sample_rate, channel_mode, channel, clip_seg_num, sliding_window, add_key=add_key, sample_key=sample_key)
        sample_rate = self.sample_rate_dict["label"]
        clip_seg_num = self.clip_seg_num_dict["label"]
        sliding_window = self.sliding_window_dict["label"]
        results = self._sample_label(results, sample_rate, clip_seg_num, sliding_window, add_key='labels', sample_key='raw_labels')
        results = self._sample_pred_label(results, sample_rate, self.pred_clip_seg_num, clip_seg_num, sliding_window, add_key='pred_labels', sample_key='raw_labels')

        return results

class VideoPredictionVideoStreamSampler(VideoStreamSampler):
    def __init__(self,
                 pred_clip_seg_num=8,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.pred_clip_seg_num = pred_clip_seg_num
    
    def _get_start_end_pred_frame_idx(self, results, sample_rate, pred_sample_num, sample_num, sliding_windows):
        start_frame = results[self.frame_idx_key] * sliding_windows + sample_num * sample_rate
        end_frame = start_frame + pred_sample_num * sample_rate
        return start_frame, end_frame
    
    def _sample_pred_label(self, results, sample_rate, pred_sample_num, sample_num, sliding_windows, add_key='pred_labels', sample_key='raw_labels'):
        container = results[sample_key]
        start_frame, end_frame = self._get_start_end_pred_frame_idx(results, sample_rate, pred_sample_num, sample_num, sliding_windows)
        frames_len = int(results['frames_len'])

        if start_frame < frames_len:
            pred_label_frames_idx = self.sample(start_frame, frames_len, sample_rate)
            pred_labels = self._labels_sample(container, start_frame=start_frame, end_frame=end_frame, samples_idx=pred_label_frames_idx, sample_rate=sample_rate).copy()
            pad_len = pred_sample_num * sample_rate - pred_labels.shape[0]
            if pad_len > 0:
                pred_labels = np.concatenate([pred_labels, np.full((pad_len), self.ignore_index)])
        else:
            pred_labels = np.full((pred_sample_num * sample_rate), self.ignore_index)

        results[add_key] = pred_labels.copy()
        return results
    
    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        for sample_key, add_key in self.sample_add_key_pair.items():
            channel_mode = self.channel_mode_dict[add_key]
            channel = self.channel_num_dict[add_key]
            sample_rate = self.sample_rate_dict[add_key]
            clip_seg_num = self.clip_seg_num_dict[add_key]
            sliding_window = self.sliding_window_dict[add_key]
            results = self._sample_frames(results, sample_rate, channel_mode, channel, clip_seg_num, sliding_window, add_key=add_key, sample_key=sample_key)
        sample_rate = self.sample_rate_dict["label"]
        clip_seg_num = self.clip_seg_num_dict["label"]
        sliding_window = self.sliding_window_dict["label"]
        results = self._sample_label(results, sample_rate, clip_seg_num, sliding_window, add_key='labels', sample_key='raw_labels')
        results = self._sample_pred_label(results, sample_rate, self.pred_clip_seg_num, clip_seg_num, sliding_window, add_key='pred_labels', sample_key='raw_labels')

        return results
