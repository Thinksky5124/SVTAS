'''
Author       : Thyssen Wen
Date         : 2023-10-17 14:52:25
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-25 11:06:04
Description  : file content
FilePath     : /SVTAS/svtas/model/losses/mse_loss.py
'''
from typing import Any, Dict
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory
from .base_loss import BaseLoss

@AbstractBuildFactory.register('loss')
class DiffusionSegmentationMSELoss(BaseLoss):
    def __init__(self,
                 loss_weight=1.0) -> None:
        super().__init__()
        self.elps = 1e-10
        self.mse_loss = nn.MSELoss(reduction='none')
        self.loss_weight = loss_weight
    
    def forward(self, model_output, input_data) -> Dict:
        # score shape [stage_num N C T]
        # masks shape [N T]
        pred_noises, noises = model_output["output"], model_output['noise']
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']

        loss = torch.mean(self.mse_loss(pred_noises * masks.unsqueeze(1), noises * masks.unsqueeze(1)).sum([1, 2])
                            / (((labels != -100).sum([1]) + self.elps) * precise_sliding_num + self.elps))
        loss = loss * self.loss_weight

        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict