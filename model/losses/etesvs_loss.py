'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-28 14:04:02
Description: loss function
FilePath: /ETESVS/model/losses/etesvs_loss.py
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES

@LOSSES.register()
class ETESVSLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 backone_loss_weight=1.0,
                 neck_loss_weight=1.0,
                 head_loss_weight=1.0,
                 smooth_weight=0.15,
                 ignore_index=-100):
        super().__init__()
        self.backone_loss_weight = backone_loss_weight
        self.neck_loss_weight = neck_loss_weight
        self.head_loss_weight = head_loss_weight
        self.smooth_weight = smooth_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate
        self.elps = 1e-10

        self.seg_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.neck_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        # self.backbone_clip_loss = TemporalSplitMeanPoolingLoss(self.num_classes, ignore_index=self.ignore_index)
        self.backbone_clip_loss = SoftLabelLoss(self.num_classes, ignore_index=self.ignore_index)
        # self.neck_frame_num_loss = TemporalClassNumMSELoss(self.num_classes, ignore_index=self.ignore_index)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, model_output, masks, labels):
        backbone_score, neck_score, head_score = model_output
        # seg_score [stage_num, N, C, T]
        # masks [N, T]
        # labels [N, T]

        # classification branch loss
        # backbone smooth label learning

        # backbone label learning
        backbone_cls_score_loss = self.backbone_clip_loss(backbone_score, labels[:, ::self.sample_rate], masks[:, ::self.sample_rate])

        # neck label learning
        neck_cls_loss = self.neck_ce(neck_score.transpose(2, 1).contiguous().view(-1, self.num_classes), labels[:, ::self.sample_rate].view(-1))
        neck_cls_score_loss = torch.sum(neck_cls_loss / (torch.sum(labels[:, ::self.sample_rate] != -100) + self.elps))
        # neck_cls_score_loss += 0.4 * self.neck_frame_num_loss(neck_score, labels[:, ::self.sample_rate], masks[:, ::self.sample_rate])
        # neck_cls_score_loss = torch.tensor(0.).cuda()

        # segmentation branch loss
        seg_loss = 0.
        for p in head_score:
            seg_cls_loss = self.seg_ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            seg_loss += torch.sum(seg_cls_loss / (torch.sum(labels != -100) + self.elps))
            seg_loss += self.smooth_weight * torch.mean(torch.clamp(
                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)
                ), min=0, max=16) * masks[:, 1:].unsqueeze(1))

        backbone_loss = self.backone_loss_weight * backbone_cls_score_loss
        neck_loss = self.neck_loss_weight * neck_cls_score_loss
        head_loss = self.head_loss_weight * seg_loss

        # output dict compose
        loss = backbone_loss + head_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["backbone_loss"] = backbone_loss
        loss_dict["neck_loss"] = neck_loss
        loss_dict["head_loss"] = head_loss
        return loss_dict

class SoftLabelLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.elps = 1e-10
    
    def forward(self, score, gt, mask):
        score = torch.mean(score, axis=-1)  # [N, num_class]
        cls_score = torch.reshape(score,
                               shape=[-1, self.num_classes])  # [N, num_class]

        # smooth label learning
        with torch.no_grad():
            device = cls_score.device
            # [N T]
            raw_labels = gt
            # deal label over num_classes
            # [N, 1]
            y = torch.zeros(raw_labels.shape, dtype=raw_labels.dtype, device=device)
            refine_label = torch.where(raw_labels != self.ignore_index, raw_labels, y)
            # [N C T]
            ce_y = F.one_hot(refine_label, num_classes=self.num_classes)

            raw_labels_repeat = torch.tile(raw_labels.unsqueeze(2), dims=[1, 1, self.num_classes])
            ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, device=device, dtype=ce_y.dtype))
            # [N C]
            smooth_label = (torch.sum(ce_y.float(), dim=1) / ce_y.shape[1])

            # [N, 1]
            x = torch.ones((smooth_label.shape[0]), device=device)
            y = torch.zeros((smooth_label.shape[0]), device=device)
            mask = torch.where(torch.sum(smooth_label, dim=1)!=0, x, y)

        cls_loss = torch.mean(self.ce(cls_score, smooth_label) * mask)
        return cls_loss

class TemporalSplitMeanPoolingLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
        self.elps = 1e-10
    
    def forward(self, x, gt, mask):
        # deal label over num_classes
        # masks [N, T]
        # gt [N, T]
        # x [N C T]

        device = gt.device
        # [N, 1]
        y = torch.zeros(gt.shape, dtype=gt.dtype, device=device)
        refine_label = torch.where(gt != self.ignore_index, gt, y)
        # [N T C]
        ce_y = F.one_hot(refine_label, num_classes=self.num_classes)
        gt_mask = ce_y.float() * mask.unsqueeze(-1)
        gt_mask = torch.permute(gt_mask, dims=[0, 2, 1])

        # batch mask
        raw_labels_repeat = torch.tile(gt.unsqueeze(2), dims=[1, 1, self.num_classes])
        ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, device=device, dtype=ce_y.dtype))
        # [N C]
        smooth_label = torch.sum(ce_y.float(), dim=1) / ce_y.shape[1]

        valid_label_num = torch.sum(torch.where(torch.sum(ce_y.float(), dim=1) != 0., torch.ones_like(smooth_label), torch.zeros_like(smooth_label)), dim=-1)

        loss_list = []
        for cls_idx in range(self.num_classes):
            mean_pool_score = torch.sum(x * gt_mask[:, cls_idx, :].unsqueeze(1), dim=-1) / (torch.sum(gt_mask[:, cls_idx, :], dim=-1) + self.elps).unsqueeze(-1)
            target = F.one_hot(torch.tensor([cls_idx] * gt.shape[0]), num_classes=self.num_classes).float().to(mean_pool_score.device)
            one = torch.ones((gt_mask[:, cls_idx, :].shape[0]), device=device)
            zero = torch.zeros((gt_mask[:, cls_idx, :].shape[0]), device=device)
            target_mask = torch.where(torch.sum(gt_mask[:, cls_idx, :], dim=-1)!=0, one, zero).unsqueeze(-1)
            clip_loss = self.ce(mean_pool_score, target * target_mask) / (valid_label_num + self.elps)
            loss_list.append(clip_loss.unsqueeze(-1))
        loss = torch.sum(torch.cat(loss_list, dim=-1), dim=-1)
        loss = torch.mean(loss)
        return loss

class TemporalClassNumMSELoss(nn.Module):
    def __init__(self,
                 num_classes,
                 ignore_index=-100):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        # self.mse = nn.MSELoss(reduction='none')
        self.mse = nn.L1Loss(reduction='none')
        self.softmax = nn.Softmax(dim=1)
        self.elps = 1e-10
    
    def forward(self, x, gt, mask):
        # masks [N, T]
        # gt [N, T]
        # x [N C T]

        x = self.softmax(x)

        device = gt.device
        # [N, 1]
        y = torch.zeros(gt.shape, dtype=gt.dtype, device=device)
        refine_label = torch.where(gt != self.ignore_index, gt, y)
        # [N T C]
        ce_y = F.one_hot(refine_label, num_classes=self.num_classes)
        gt_mask = ce_y.float() * mask.unsqueeze(-1)
        gt_mask = torch.permute(gt_mask, dims=[0, 2, 1])

        # batch mask
        raw_labels_repeat = torch.tile(gt.unsqueeze(2), dims=[1, 1, self.num_classes])
        ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, device=device, dtype=ce_y.dtype))
        # [N C]
        smooth_label = torch.sum(ce_y.float(), dim=1) / ce_y.shape[1]

        valid_label_num = torch.sum(torch.where(torch.sum(ce_y.float(), dim=1) != 0., torch.ones_like(smooth_label), torch.zeros_like(smooth_label)), dim=-1)

        loss_list = []
        for cls_idx in range(self.num_classes):
            cls_frame_num = torch.sum(x * gt_mask[:, cls_idx, :].unsqueeze(1), dim=-1)
            target = (torch.sum(gt_mask[:, cls_idx, :], dim=-1) + self.elps).unsqueeze(-1)
            target_mask = F.one_hot(torch.tensor([cls_idx] * gt.shape[0]), num_classes=self.num_classes).float().to(cls_frame_num.device)
            clip_loss = self.mse(cls_frame_num * target_mask, target * target_mask) / (valid_label_num + self.elps).unsqueeze(-1)
            clip_loss = torch.sum(clip_loss, dim=-1)
            loss_list.append(clip_loss.unsqueeze(-1))
        loss = torch.sum(torch.cat(loss_list, dim=-1), dim=-1)
        loss = torch.mean(loss)
        return loss
            
class FocalLoss(nn.Module):
    def __init__(self, num_classes, alpha=0.25, gamma=2, size_average=True):
        super(FocalLoss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha)==num_classes   
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha)

        self.gamma = gamma

    def forward(self, preds, labels):
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds, dim=1) 
        preds_logsoft = torch.log(preds_softmax)
        
        #focal_loss func, Loss = -α(1-yi)**γ *ce_loss(xi,yi)
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))
        self.alpha = self.alpha.gather(0,labels.view(-1))
        # torch.pow((1-preds_softmax), self.gamma) is (1-pt)**γ
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 

        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
