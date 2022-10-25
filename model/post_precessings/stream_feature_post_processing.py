'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 15:30:48
Description: model postprecessing
FilePath     : /SVTAS/model/post_precessings/stream_feature_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class StreamFeaturePostProcessing():
    def __init__(self,
                 sliding_window,
                 ignore_index=-100):
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.init_flag = False
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_feature = []
        self.video_gt = []
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        with torch.no_grad():
            if torch.is_tensor(seg_scores):
                self.pred_feature.append(seg_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
            else:
                self.pred_feature.append(seg_scores[-1, :, :, 0:self.sliding_window].copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].copy())


    def output(self):
        pred_feature_list = []
        video_gt = np.concatenate(self.video_gt, axis=1)
        pred_feature = np.concatenate(self.pred_feature, axis=2)
        
        for bs in range(pred_feature.shape[0]):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            feature = pred_feature[bs, :, :ignore_start]
            feature = feature.squeeze()
            pred_feature_list.append(feature.copy())

        return pred_feature_list