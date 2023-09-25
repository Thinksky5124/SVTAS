'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:14:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 21:17:49
Description  : file content
FilePath     : /SVTAS/svtas/model/losses/asrf_loss.py
'''
# https://github.com/yiskw713/asrf/libs/loss_fn/__init__.py
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from svtas.utils import AbstractBuildFactory

class GaussianSimilarityTMSE(nn.Module):
    """
    Temporal MSE Loss Function with Gaussian Similarity Weighting
    """

    def __init__(self, sample_rate=1, threshold=4, sigma=1.0, ignore_index=255):
        super().__init__()
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.ignore_index = ignore_index
        self.mse = nn.MSELoss(reduction="none")
        self.sigma = sigma
        self.elps = 1e-10

    def forward(self, preds, gts, sim_index, masks, precise_sliding_num):
        """
        Args:
            preds: the output of model before softmax. (N, C, T)
            gts: Ground Truth. (N, T)
            sim_index: similarity index. (N, C, T)
        Return:
            the value of Temporal MSE weighted by Gaussian Similarity.
        """
        total_loss = 0.0
        batch_size = preds.shape[0]
        sim_index = F.interpolate(
            input=sim_index.unsqueeze(1),
            scale_factor=[1, self.sample_rate],
            mode="nearest").squeeze(1)
        for pred, gt, sim, b, p_num in zip(preds, gts, sim_index, range(batch_size), precise_sliding_num):
            pred = pred[:, torch.where(gt != self.ignore_index)[0]]
            sim = sim[:, torch.where(gt != self.ignore_index)[0]]
            mask = masks[b, torch.where(gt != self.ignore_index)[0]]

            # calculate gaussian similarity
            diff = sim[:, 1:] - sim[:, :-1]
            similarity = torch.exp(
                (-1 * torch.norm(diff, dim=0)) / (2 * self.sigma**2))

            # calculate temporal mse
            loss = torch.clamp(
                self.mse(F.log_softmax(pred[:, 1:], dim=1), F.log_softmax(pred.detach()[:, :-1], dim=1))
                , min=0, max=self.threshold**2) * mask[1:].unsqueeze(0)

            # gaussian similarity weighting
            loss = torch.mean(similarity * loss, dim=-1) 

            total_loss += torch.mean(loss) / (p_num + self.elps)

        return total_loss / batch_size


class ActionSegmentationLoss(nn.Module):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
    """

    def __init__(self,
                 num_classes,
                 sample_rate=1,
                 class_weight=None,
                 threshold=4.,
                 ignore_index=255,
                 ce_weight=1.0,
                 gstmse_weight=0.15):
        super().__init__()
        self.criterions = nn.ModuleList()
        self.weights = []

        self.num_classes = num_classes

        self.criterions.append(
            nn.CrossEntropyLoss(weight=torch.Tensor(class_weight),
                                ignore_index=ignore_index,
                                reduction='none'))
        self.weights.append(ce_weight)

        self.criterions.append(
            GaussianSimilarityTMSE(sample_rate=sample_rate,
                                    threshold=threshold,
                                    ignore_index=ignore_index))
        self.weights.append(gstmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)
        self.elps = 1e-10

    def forward(self, preds, labels, sim_index, masks, precise_sliding_num):
        """
        Args:
            preds: torch.float (N, C, T).
            labels: torch.int64 (N, T).
            sim_index: torch.float (N, C', T).
        """
        loss = 0.0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, GaussianSimilarityTMSE):
                loss += weight * criterion(preds, labels, sim_index, masks, precise_sliding_num)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                b, _, t = preds.shape
                seg_cls_loss = criterion(preds.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
                loss += torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
            else:
                loss += weight * criterion(preds, labels)

        return loss


class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(self, pos_weight):
        super().__init__()

        self.criterions = nn.ModuleList()

        self.criterions.append(
            nn.BCEWithLogitsLoss(pos_weight=torch.Tensor(pos_weight), reduction='none'))

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)
        self.elps = 1e-10

    def forward(self, preds, gts, masks, precise_sliding_num):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch.float (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                boundary_loss = criterion(pred * mask.unsqueeze(0), gt * mask.unsqueeze(0))
                loss += torch.mean(torch.mean(boundary_loss, dim=-1) / (precise_sliding_num + self.elps))

        return loss / batch_size


@AbstractBuildFactory.register('loss')
class ASRFLoss(nn.Module):

    def __init__(self,
                 num_classes,
                 class_weight,
                 pos_weight,
                 sample_rate=1,
                 boundary_loss_weight=0.1,
                 threshold=4.,
                 ignore_index=-100,
                 ce_weight=1.0,
                 gstmse_weight=1.0):
        super().__init__()
        self.criterion_cls = ActionSegmentationLoss(sample_rate=sample_rate,
                                                    threshold=threshold,
                                                    class_weight=class_weight,
                                                    ignore_index=ignore_index,
                                                    ce_weight=ce_weight,
                                                    gstmse_weight=gstmse_weight,
                                                    num_classes=num_classes)
        self.criterion_boundary = BoundaryRegressionLoss(pos_weight=pos_weight)
        self.boundary_loss_weight = boundary_loss_weight
        self.sample_rate = sample_rate

    def get_boundary(self, labels):
        # labels [N, T]
        # define the frame where new action starts as boundary frame
        boundary = torch.zeros_like(labels)
        for gt, b in zip(labels, range(labels.shape[0])):
            last = labels[b, 0]
            boundary[b, 0] = 1
            for i in range(1, gt.shape[0]):
                if last != gt[i]:
                    boundary[b, i] = 1
                    last = gt[i]
        return boundary

    def forward(self, model_output, input_data):
        head_score = model_output["output"]
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        
        output_cls = head_score["cls"]
        outputs_boundary = head_score["boundary"]
        features = head_score["features"]

        boundary = self.get_boundary(labels=labels)
        num_stages = output_cls.shape[0]
        cls_loss = 0.0
        for out in output_cls:
            cls_loss += self.criterion_cls(out, labels, features, masks, precise_sliding_num)
        loss = cls_loss / num_stages
        boundary_loss = 0.0
        for out in outputs_boundary:
            boundary_loss += self.boundary_loss_weight * self.criterion_boundary(
                out, boundary, masks, precise_sliding_num)
        loss = loss + boundary_loss / num_stages
        loss_dict={}
        loss_dict["loss"] = loss
        return loss_dict