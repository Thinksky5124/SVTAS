'''
Author       : Thyssen Wen
Date         : 2022-05-26 18:50:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 21:16:23
Description  : Score Post precessing Module
FilePath     : /ETESVS/model/post_precessings/score_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class ScorePostProcessing():
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_scores = None
        self.video_gt = None
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        with torch.no_grad():
            self.pred_scores = seg_scores[-1, :].detach().cpu().numpy().copy()
            self.video_gt = gt.detach().cpu().numpy().copy()
            pred = np.argmax(seg_scores[-1, :].detach().cpu().numpy(), axis=-2)
            acc = np.mean((np.sum(pred == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
        return acc

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []

        for bs in range(self.pred_scores.shape[0]):
            index = np.where(self.video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [self.pred_scores[bs].shape[-1]])
            predicted = np.argmax(self.pred_scores[bs, :, :ignore_start], axis=0)
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list