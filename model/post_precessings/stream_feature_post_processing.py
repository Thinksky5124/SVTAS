'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 18:49:49
Description: model postprecessing
FilePath     : /ETESVS/model/post_precessings/feature_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class StreamFeaturePostProcessing():
    def __init__(self,
                 feature_dim,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 ignore_index=-100):
        self.feature_dim = feature_dim
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.init_flag = False
    
    def init_scores(self, sliding_num, batch_size):
        max_temporal_len = sliding_num * self.sliding_window + self.sample_rate * self.clip_seg_num
        sample_videos_max_len = max_temporal_len + \
            ((self.clip_seg_num * self.sample_rate) - max_temporal_len % (self.clip_seg_num * self.sample_rate))
        if sample_videos_max_len % self.sliding_window != 0:
            sample_videos_max_len = sample_videos_max_len + \
                (self.sliding_window - (sample_videos_max_len % self.sliding_window))
        self.sample_videos_max_len = sample_videos_max_len
        self.pred_feature = np.zeros((batch_size, self.feature_dim, sample_videos_max_len))
        self.video_gt = np.full((batch_size, sample_videos_max_len), self.ignore_index)
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        with torch.no_grad():
            start_frame = idx * self.sliding_window
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + (self.clip_seg_num * self.sample_rate)
            self.pred_feature[:, :, start_frame:end_frame] = seg_scores[-1, :].detach().cpu().numpy().copy()
            self.video_gt[:, start_frame:end_frame] = gt.detach().cpu().numpy().copy()

    def output(self):
        pred_feature_list = []
        
        for bs in range(self.pred_feature.shape[0]):
            index = np.where(self.video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [self.sample_videos_max_len])
            feature = self.pred_feature[bs, :, :ignore_start]
            feature = feature.squeeze()
            pred_feature_list.append(feature.copy())

        return pred_feature_list