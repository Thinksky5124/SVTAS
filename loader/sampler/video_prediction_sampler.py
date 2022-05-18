'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:58:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 16:47:30
Description  : Video Prediction Sampler
FilePath     : /ETESVS/loader/sampler/video_prediction_sampler.py
'''
import numpy as np
from ..builder import SAMPLER
from .feature_sampler import FeatureStreamSampler
from .frame_sampler import VideoFrameSample
from PIL import Image

@SAMPLER.register()
class VideoPredictionFeatureStreamSampler(FeatureStreamSampler):
    def __init__(self,
                 pred_clip_seg_num=8,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.pred_clip_seg_num = pred_clip_seg_num
    
    def _pred_label_sample(self, labels, start_pred_idx, frames_len):
        pred_end_idx = start_pred_idx + self.pred_clip_seg_num * self.sample_rate
        if start_pred_idx < frames_len and pred_end_idx < frames_len:
            pred_label_frames_idx = self.sample(start_pred_idx, pred_end_idx, self.sample_rate)
            pred_labels = self._labels_sample(labels, start_frame=start_pred_idx, end_frame=pred_end_idx, samples_idx=pred_label_frames_idx).copy()
        elif start_pred_idx < frames_len and pred_end_idx >= frames_len:
            pred_label_frames_idx = self.sample(start_pred_idx, frames_len, self.sample_rate)
            pred_labels = self._labels_sample(labels, start_frame=start_pred_idx, end_frame=frames_len, samples_idx=pred_label_frames_idx).copy()
            pad_len = self.pred_clip_seg_num * self.sample_rate - pred_labels.shape[0]
            pred_labels = np.concatenate([pred_labels, np.full((pad_len), self.ignore_index)])
        else:
            pred_labels = np.full((self.pred_clip_seg_num * self.sample_rate), self.ignore_index)
        return pred_labels
    
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
        pred_labels = self._pred_label_sample(labels, end_frame, frames_len)

        if start_frame < frames_len and end_frame < frames_len:
            frames_feature, labels, mask = self._all_valid_frames(start_frame, end_frame, feature_len, feature, labels)
        elif start_frame < frames_len and end_frame >= frames_len:
            frames_feature, labels, mask = self._some_valid_frames(start_frame, end_frame, feature_len, frames_len, feature, labels)
        else:
            frames_feature = np.zeros((2048, self.clip_seg_num))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['feature'] = frames_feature[:, :self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        results['pred_labels'] = pred_labels.copy()
        return results

class VideoPredictionVideoStreamSampler(VideoFrameSample):
    def __init__(self,
                 pred_clip_seg_num=8,
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.pred_clip_seg_num = pred_clip_seg_num
    
    def _pred_label_sample(self, labels, start_pred_idx, frames_len):
        pred_end_idx = start_pred_idx + self.pred_clip_seg_num * self.sample_rate
        if start_pred_idx < frames_len and pred_end_idx < frames_len:
            pred_label_frames_idx = self.sample(start_pred_idx, pred_end_idx, self.sample_rate)
            pred_labels = self._labels_sample(labels, start_frame=start_pred_idx, end_frame=pred_end_idx, samples_idx=pred_label_frames_idx).copy()
        elif start_pred_idx < frames_len and pred_end_idx >= frames_len:
            pred_label_frames_idx = self.sample(start_pred_idx, frames_len, self.sample_rate)
            pred_labels = self._labels_sample(labels, start_frame=start_pred_idx, end_frame=frames_len, samples_idx=pred_label_frames_idx).copy()
            pad_len = self.pred_clip_seg_num * self.sample_rate - pred_labels.shape[0]
            pred_labels = np.concatenate([pred_labels, np.full((pad_len), self.ignore_index)])
        else:
            pred_labels = np.full((self.pred_clip_seg_num * self.sample_rate), self.ignore_index)
        return pred_labels
    
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
        pred_labels = self._pred_label_sample(labels, end_frame, frames_len)
        
        if start_frame < frames_len and end_frame < frames_len:
            imgs, labels, mask = self._all_valid_frames(start_frame, end_frame, video_len, container, labels)
        elif start_frame < frames_len and end_frame >= frames_len:
            imgs, labels, mask = self._some_valid_frames(start_frame, end_frame, video_len, frames_len, container, labels)
        else:
            imgs = []
            np_frames = np.zeros((240, 320, 3))
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        results['pred_labels'] = pred_labels.copy()
        return results
