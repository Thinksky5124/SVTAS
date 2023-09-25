'''
Author       : Thyssen Wen
Date         : 2022-05-26 18:50:50
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 16:46:32
Description  : Score Post precessing Module
FilePath     : /SVTAS/svtas/model/post_precessings/score_post_processing.py
'''
import numpy as np
import torch
from svtas.utils import AbstractBuildFactory
from . import refine_method

@AbstractBuildFactory.register('post_precessing')
class ScorePostProcessing():
    def __init__(self,
                 ignore_index=-100):
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_scores = None
        self.video_gt = None
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        with torch.no_grad():
            if torch.is_tensor(seg_scores):
                self.pred_scores = seg_scores[-1, :].detach().cpu().numpy().copy()
                self.video_gt = gt.detach().cpu().numpy().copy()
                pred = np.argmax(seg_scores[-1, :].detach().cpu().numpy(), axis=-2)
                acc = np.mean((np.sum(pred == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
            else:
                self.pred_scores = seg_scores[-1, :].copy()
                self.video_gt = gt.copy()
                pred = np.argmax(seg_scores[-1, :], axis=-2)
                acc = np.mean((np.sum(pred == gt, axis=1) / (np.sum(gt != self.ignore_index, axis=1) + self.epls)))
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

@AbstractBuildFactory.register('post_precessing')
class ScorePostProcessingWithRefine(ScorePostProcessing):
    def __init__(self,
                 refine_method_cfg: dict = None,
                 ignore_index=-100):
        super().__init__(ignore_index)
        name = refine_method_cfg.pop("name")
        self.post_process_meth = getattr(refine_method, name)(**refine_method_cfg)
        self.pred_cls = None
    
    def init_scores(self, sliding_num, batch_size):
        self.pred_cls = None
        return super().init_scores(sliding_num, batch_size)
    
    def update(self, seg_scores, gt, idx):
        cls_scores = seg_scores['cls']
        boundary_scores = seg_scores['boundary']
        with torch.no_grad():
            cls_scores_np = cls_scores[-1, :].detach().cpu().numpy().copy()
            boundary_np = boundary_scores[-1, :].detach().cpu().numpy().copy()
            self.pred_scores = cls_scores_np.copy()
            self.video_gt = gt.detach().cpu().numpy().copy()
            pred_cls = self.post_process_meth(cls_scores_np, boundary_np).detach().cpu().numpy()
            self.pred_cls = pred_cls
            acc = np.mean((np.sum(pred_cls == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
        return acc

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []

        for bs in range(self.pred_scores.shape[0]):
            index = np.where(self.video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [self.pred_scores[bs].shape[-1]])
            predicted = self.pred_cls[bs, :ignore_start]
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list