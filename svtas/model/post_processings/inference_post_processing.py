'''
Author       : Thyssen Wen
Date         : 2023-10-30 16:30:30
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 19:47:57
Description  : file content
FilePath     : \ETESVS\svtas\model\post_processings\inference_post_processing.py
'''
import numpy as np
import torch
from svtas.utils import AbstractBuildFactory
from . import refine_method
from .base_post_processing import BasePostProcessing

@AbstractBuildFactory.register('post_processing')
class InferencePostProcessing(BasePostProcessing):
    def __init__(self):
        super().__init__()
        self.epls = 1e-10
    
    def init_scores(self, sliding_num=None, batch_size=None):
        self.pred_scores = None
        self.init_flag = True

    def update(self, seg_scores, gt=None, idx=None):
        # seg_scores [stage_num N C T]
        # gt [N T]
        self.pred_scores = seg_scores[-1, :].copy()
        return 0

    def output(self):
        pred_score_list = []
        pred_cls_list = []

        for bs in range(self.pred_scores.shape[0]):
            predicted = np.argmax(self.pred_scores[bs], axis=0)
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs].copy())

        return pred_score_list, pred_cls_list