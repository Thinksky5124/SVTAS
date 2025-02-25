'''
Author: Thyssen Wen
Date: 2022-04-27 20:01:21
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-19 10:30:24
Description: MS-TCN loss model
FilePath     : /ETESVS/model/losses/segmentation_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision 
import json 

from ..builder import LOSSES

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        return loss 
        import pdb; pdb.set_trace()
        if self.size_average: return loss.mean()
        else: return loss.sum()


@LOSSES.register()
class SegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 loss_weight=1.0,
                 sample_rate=1,
                 smooth_weight=0.5,
                 ignore_index=-100,
                 class_weight=None):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10
        self.loss_weight = loss_weight
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)
            self.ce = nn.CrossEntropyLoss(weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.mse = nn.MSELoss(reduction='none')#
        # self.focal_loss = FocalLoss(gamma=2)#torchvision.ops.focal_loss.sigmoid_focal_loss#
        self.iter = {'inp': None, 'lbl':None, 'loss':None}
    def focal_loss(self, predictions, targets, ignore_idx=None):
        """
        Compute Focal Loss.
        
        Args:
        - predictions: Tensor of shape (N, C), probabilities (after softmax).
        - targets: Tensor of shape (N, C), one-hot encoded.
        - alpha: Weighting factor for class imbalance.
        - gamma: Focusing parameter for hard-to-classify examples.
        
        Returns:
        - Loss: Scalar loss value.
        """
        # Avoid zero probabilities (numerical stability)
        alpha=1
        gamma=2
        predictions = F.softmax(predictions.clamp(min=1e-7, max=1-1e-7),dim=1)

        if ignore_idx is not None:
            targets = targets.clone()  # Avoid modifying the original targets
            targets[ignore_idx, :] = 0  # Set the target class at ignore_idx to 0 for all samples

        # Focal Loss formula
        focal_loss = -alpha * (1 - predictions) ** gamma * targets * predictions.log()
        return focal_loss.mean(dim=1)

    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        import pdb; pdb.set_trace()
        head_score = model_output
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        _, b, _, t = head_score.shape

        loss = 0.
        for p in head_score:


                #### focal loss #####
            # inp = p.transpose(2, 1).contiguous().view(-1, self.num_classes)
            # lbl = labels.view(-1)
            # lbl = lbl.clamp(0, self.num_classes - 1)
            # assert lbl.min() >= 0 and lbl.max() < self.num_classes, "Labels out of range"
            # assert lbl.dtype == torch.int64, "Labels must be int64"
            # lbl_one_hot = F.one_hot(lbl, num_classes=5)
            # # self.iter['inp'] = inp 
            # # self.iter['lbl'] = lbl 
            # seg_cls_loss = self.focal_loss(inp, lbl_one_hot, self.ignore_index)
            # if torch.isnan(seg_cls_loss).any() or torch.isinf(seg_cls_loss).any():
            #     print("NaN or Inf detected")
            
            # json.dump(self.iter, open('recent.json', 'w'))

            # seg_cls_loss = torch.mean(self.focal_loss(inp.float(), lbl_one_hot.float()), dim=1)
            
            #### ce loss ####
            seg_cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            loss += torch.sum(torch.sum(torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1) / (precise_sliding_num + self.elps)) / (torch.sum(labels != -100) + self.elps)
            loss += self.smooth_weight * torch.mean((torch.mean(torch.reshape(torch.clamp(    self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1))    , min=0, max=16) * masks[:, 1:].unsqueeze(1), [b, -1]), dim=-1) / (precise_sliding_num + self.elps)))
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict
    

@LOSSES.register()
class SegmentationLossMultiLabel(nn.Module):
    def __init__(self,
                 num_classes_action,
                 num_classes_branches,
                 loss_weight=1.0,
                 sample_rate=1,
                 smooth_weight=0.5,
                 ignore_index=-100,
                 class_weight=None):
        super().__init__()
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes_action = num_classes_action
        self.num_classes_branches = num_classes_branches
        self.sample_rate = sample_rate
        self.elps = 1e-10
        self.loss_weight = loss_weight
        if class_weight is not None:
            class_weight = torch.tensor(class_weight)
            self.ce = nn.CrossEntropyLoss(weight=class_weight, ignore_index=self.ignore_index, reduction='none')
        else:
            self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.mse = nn.MSELoss(reduction='none')#
        # self.focal_loss = FocalLoss(gamma=2)#torchvision.ops.focal_loss.sigmoid_focal_loss#
        self.iter = {'inp': None, 'lbl':None, 'loss':None}
    def focal_loss(self, predictions, targets, num_classes, ignore_idx=None):
        """
        Compute Focal Loss.
        
        Args:
        - predictions: Tensor of shape (N, C), probabilities (after softmax).
        - targets: Tensor of shape (N, C), one-hot encoded.
        - alpha: Weighting factor for class imbalance.
        - gamma: Focusing parameter for hard-to-classify examples.
        
        Returns:
        - Loss: Scalar loss value.
        """
        # Avoid zero probabilities (numerical stability)
        alpha=1
        gamma=2
        predictions = F.softmax(predictions.clamp(min=1e-7, max=1-1e-7),dim=1)

        if ignore_idx is not None:
            targets = targets.clone()  # Avoid modifying the original targets
            targets[ignore_idx, :] = 0  # Set the target class at ignore_idx to 0 for all samples

        # Focal Loss formula
        focal_loss = -alpha * (1 - predictions) ** gamma * targets * predictions.log()
        return focal_loss.mean(dim=1)
    
    def compute_loss(self, model_output, labels, input_data, num_classes):
        """
        Compute the loss for a given model output and input data.

        Parameters:
            model_output (torch.Tensor): Output from the model.
            input_data (dict): Dictionary containing "masks", "labels", and "precise_sliding_num".

        Returns:
            torch.Tensor: The computed loss value.
        """
        head_score = model_output
        labels = labels 
        masks, precise_sliding_num = input_data["masks"], input_data["precise_sliding_num"]
        _, b, _, t = head_score.shape

        loss = 0.0

        for p in head_score:
            # Cross-entropy loss
            seg_cls_loss = self.ce(
                p.transpose(2, 1).contiguous().view(-1, num_classes), 
                labels.view(-1)
            )
            loss += torch.sum(
                torch.sum(
                    torch.reshape(seg_cls_loss, shape=[b, t]), dim=-1
                ) / (precise_sliding_num + self.elps)
            ) / (torch.sum(labels != -100) + self.elps)

            # Smoothness loss
            smooth_loss = self.mse(
                F.log_softmax(p[:, :, 1:], dim=1), 
                F.log_softmax(p.detach()[:, :, :-1], dim=1)
            )
            smooth_loss = torch.clamp(smooth_loss, min=0, max=16) * masks[:, 1:].unsqueeze(1)
            smooth_loss = torch.mean(
                torch.mean(
                    torch.reshape(smooth_loss, [b, -1]), dim=-1
                ) / (precise_sliding_num + self.elps)
            )
            loss += self.smooth_weight * smooth_loss

        return loss

    def forward(self, model_output, input_data):
        # score shape [stage_num N C T]
        # masks shape [N T]
        action_loss = self.compute_loss(model_output['action_score'], input_data['labels'], input_data, self.num_classes_action)
        branch_loss = self.compute_loss(model_output['branch_score'], input_data['branch_labels'], input_data, self.num_classes_branches)
        
        loss_dict={}
        loss_dict["action_loss"] = action_loss * self.loss_weight
        loss_dict["branch_loss"] = branch_loss * self.loss_weight
        loss_dict["loss"] = action_loss + branch_loss
        return loss_dict