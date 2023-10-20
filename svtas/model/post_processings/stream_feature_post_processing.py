'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-20 23:06:12
Description: model postprecessing
FilePath     : /SVTAS/svtas/model/post_processings/stream_feature_post_processing.py
'''
import numpy as np
import torch
from svtas.utils import AbstractBuildFactory
from svtas.utils.fileio import NPYStreamWriter
from .base_post_processing import BasePostProcessing

@AbstractBuildFactory.register('post_processing')
class StreamFeaturePostProcessing(BasePostProcessing):
    def __init__(self,
                 sliding_window=None,
                 ignore_index=-100):
        super().__init__()
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_feature = []
        self.video_gt = []
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        with torch.no_grad():
            if torch.is_tensor(seg_scores):
                for bs in range(seg_scores.shape[1]):
                    if len(self.pred_feature) < (bs + 1):
                        self.pred_feature.append(NPYStreamWriter())
                    if self.sliding_window:
                        self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                    else:
                        self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, :].detach().cpu().numpy().copy())
                if self.sliding_window:
                    self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
                else:
                    self.video_gt.append(gt[:, :].detach().cpu().numpy().copy())
            else:
                for bs in range(seg_scores.shape[1]):
                    if len(self.pred_feature) < (bs + 1):
                        self.pred_feature.append(NPYStreamWriter())
                    if self.sliding_window:
                        self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, 0:self.sliding_window].copy())
                    else:
                        self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, :].copy())
                if self.sliding_window:
                    self.video_gt.append(gt[:, 0:self.sliding_window].copy())
                else:
                    self.video_gt.append(gt[:, :].copy())


    def output(self):
        pred_feature_list = []
        video_gt = np.concatenate(self.video_gt, axis=1)
        
        for bs in range(video_gt.shape[0]):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            self.pred_feature[bs].dump()
            pred_feature_list.append({"writer":self.pred_feature[bs], "len":ignore_start})

        return pred_feature_list