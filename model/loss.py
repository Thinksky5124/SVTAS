import torch
import torch.nn as nn
import torch.nn.functional as F

class ETETSLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 seg_weight=1.0,
                 cls_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.sample_rate = sample_rate

        self.seg_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.cls_ce = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, seg_score, cls_score, masks, labels):
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        # masks [N, T]
        # labels [N, T]
        # segmentation branch loss
        seg_loss = 0.
        for p in seg_score:
            seg_loss += self.seg_ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), labels.view(-1))
            seg_loss += 0.15 * torch.mean(torch.clamp(
                self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)
                ), min=0, max=16) * masks[:, 1:].unsqueeze(1))

        # classification branch loss
        # [N T]
        ce_y = labels[:, ::self.sample_rate]
        cls_loss = self.cls_ce(cls_score.transpose(2, 1).contiguous().view(-1, self.num_classes), ce_y.view(-1))

        cls_loss = self.cls_weight * cls_loss
        seg_loss = self.seg_weight * seg_loss
        return cls_loss, seg_loss