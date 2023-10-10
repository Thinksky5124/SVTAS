'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 19:37:37
Description: model postprecessing
FilePath     : /SVTAS/svtas/model/post_processings/stream_score_post_processing.py
'''
import numpy as np
import torch
from svtas.utils import AbstractBuildFactory
from . import refine_method
from .base_post_processing import BasePostProcessing

@AbstractBuildFactory.register('post_processing')
class StreamScorePostProcessing(BasePostProcessing):
    def __init__(self,
                 sliding_window=None,
                 ignore_index=-100):
        super().__init__()
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
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
                if self.sliding_window:
                    self.pred_scores.append(seg_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy())
                    self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
                    pred = np.argmax(seg_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy(), axis=-2)
                    acc = np.mean((np.sum(pred == gt[:, 0:self.sliding_window].detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
                else:
                    self.pred_scores.append(seg_scores[-1, :, :, :].detach().cpu().numpy().copy())
                    self.video_gt.append(gt[:, :].detach().cpu().numpy().copy())
                    pred = np.argmax(seg_scores[-1, :, :, :].detach().cpu().numpy(), axis=-2)
                    acc = np.mean((np.sum(pred == gt[:, :].detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
            else:
                if self.sliding_window:
                    self.pred_scores.append(seg_scores[-1, :, :, 0:self.sliding_window].copy())
                    self.video_gt.append(gt[:, 0:self.sliding_window].copy())
                    pred = np.argmax(seg_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy(), axis=-2)
                    acc = np.mean((np.sum(pred == gt[:, 0:self.sliding_window].detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
                else:
                    self.pred_scores.append(seg_scores[-1, :, :, :].copy())
                    self.video_gt.append(gt[:, :].copy())
                    pred = np.argmax(seg_scores[-1, :, :, :].detach().cpu().numpy(), axis=-2)
                    acc = np.mean((np.sum(pred == gt[:, :].detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
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

@AbstractBuildFactory.register('post_processing')
class StreamScorePostProcessingWithRefine(StreamScorePostProcessing):
    def __init__(self,
                 sliding_window,
                 refine_method_cfg: dict = None,
                 ignore_index=-100):
        super().__init__(sliding_window, ignore_index)
        name = refine_method_cfg.pop("name")
        self.post_process_meth = getattr(refine_method, name)(**refine_method_cfg)
        self.pred_cls = []
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_cls = []
        return super().init_scores(sliding_num, batch_size)

    def update(self, seg_scores, gt, idx):
        cls_scores = seg_scores['cls']
        boundary_scores = seg_scores['boundary']
        with torch.no_grad():
            self.pred_scores.append(cls_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy())
            self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
            cls_scores_np = cls_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy()
            boundary_np = boundary_scores[-1, :, :, 0:self.sliding_window].detach().cpu().numpy().copy()
            pred_cls = self.post_process_meth(cls_scores_np, boundary_np).detach().cpu().numpy()
            self.pred_cls.append(pred_cls)
            acc = np.mean((np.sum(pred_cls == gt[:, 0:self.sliding_window].detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
        return acc
    
    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []
        video_gt = np.concatenate(self.video_gt, axis=1)
        pred_scores = np.concatenate(self.pred_scores, axis=2)
        pred_cls = np.concatenate(self.pred_cls, axis=1)

        for bs in range(pred_scores.shape[0]):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            predicted = pred_cls[bs, :ignore_start]
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list