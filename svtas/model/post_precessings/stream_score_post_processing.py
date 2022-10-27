'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 15:31:09
Description: model postprecessing
FilePath     : /SVTAS/model/post_precessings/stream_score_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class StreamScorePostProcessing():
    def __init__(self,
                 sliding_window,
                 ignore_index=-100):
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_scores = []
        self.video_gt = []
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        with torch.no_grad():
            if torch.is_tensor(seg_scores):
                self.pred_scores.append(seg_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
                pred = np.argmax(seg_scores[-1, :].detach().cpu().numpy(), axis=-2)
                acc = np.mean((np.sum(pred == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
            else:
                self.pred_scores.append(seg_scores[-1, :, :, 0:self.sliding_window].copy())
                self.video_gt.append(gt[:, 0:self.sliding_window].copy())
                pred = np.argmax(seg_scores[-1, :], axis=-2)
                acc = np.mean((np.sum(pred == gt, axis=1) / (np.sum(gt != self.ignore_index, axis=1) + self.epls)))
        return acc

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []
        video_gt = np.concatenate(self.video_gt, axis=1)
        pred_scores = np.concatenate(self.pred_scores, axis=2)

        for bs in range(pred_scores.shape[0]):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            predicted = np.argmax(pred_scores[bs, :, :ignore_start], axis=0)
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list