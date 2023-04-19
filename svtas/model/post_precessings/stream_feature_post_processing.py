'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-11 20:39:46
Description: model postprecessing
FilePath     : /SVTAS/svtas/model/post_precessings/stream_feature_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING
from ...utils.stream_writer import NPYStreamWriter

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
                for bs in range(seg_scores.shape[1]):
                    if len(self.pred_feature) < (bs + 1):
                        self.pred_feature.append(NPYStreamWriter())
                    self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
            else:
                for bs in range(seg_scores.shape[1]):
                    if len(self.pred_feature) < (bs + 1):
                        self.pred_feature.append(NPYStreamWriter())
                    self.pred_feature[bs].stream_write(seg_scores[-1, bs, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].copy())


    def output(self):
        pred_feature_list = []
        video_gt = np.concatenate(self.video_gt, axis=1)
        
        for bs in range(video_gt.shape[0]):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            self.pred_feature[bs].dump()
            pred_feature_list.append({"writer":self.pred_feature[bs], "len":ignore_start})

        return pred_feature_list