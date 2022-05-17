'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-17 19:11:38
Description: model postprecessing
FilePath     : /ETESVS/model/post_precessings/score_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class ScorePostProcessing():
    def __init__(self,
                 num_classes,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 ignore_index=-100):
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
    
    def init_scores(self, sliding_num, batch_size):
        max_temporal_len = sliding_num * self.sliding_window + self.sample_rate * self.clip_seg_num
        sample_videos_max_len = max_temporal_len + \
            ((self.clip_seg_num * self.sample_rate) - max_temporal_len % (self.clip_seg_num * self.sample_rate))
        if sample_videos_max_len % self.sliding_window != 0:
            sample_videos_max_len = sample_videos_max_len + \
                (self.sliding_window - (sample_videos_max_len % self.sliding_window))
        self.sample_videos_max_len = sample_videos_max_len
        self.pred_scores = np.zeros((batch_size, self.num_classes, sample_videos_max_len))
        self.video_gt = np.full((batch_size, sample_videos_max_len), self.ignore_index)
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        with torch.no_grad():
            start_frame = idx * self.sliding_window
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + (self.clip_seg_num * self.sample_rate)
            self.pred_scores[:, :, start_frame:end_frame] = seg_scores[-1, :].detach().cpu().numpy().copy()
            self.video_gt[:, start_frame:end_frame] = gt.detach().cpu().numpy().copy()
            pred = np.argmax(seg_scores[-1, :].detach().cpu().numpy(), axis=-2)
            acc = np.mean((np.sum(pred == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
        return acc

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []

        for bs in range(self.pred_scores.shape[0]):
            index = np.where(self.video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [self.sample_videos_max_len])
            predicted = np.argmax(self.pred_scores[bs, :, :ignore_start], axis=0)
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list