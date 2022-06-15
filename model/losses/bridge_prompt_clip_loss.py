'''
Author       : Thyssen Wen
Date         : 2022-06-15 19:43:47
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 20:48:56
Description  : Bridge Prompt CLIP Loss ref:https://github.com/ttlmh/Bridge-Prompt/blob/master/train.py
FilePath     : /ETESVS/model/losses/bridge_prompt_clip_loss.py
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .segmentation_loss import SegmentationLoss

from ..builder import LOSSES

@LOSSES.register()
class BridgePromptCLIPSegmentationLoss(nn.Module):
    def __init__(self,
                 num_classes,
                 sample_rate=4,
                 smooth_weight=0.15,
                 img_seg_loss_weights=1.0,
                 clip_loss_weight=1.0,
                 ignore_index=-100):
        super().__init__()
        self.ignore_index = ignore_index
        self.num_classes = num_classes
        self.smooth_weight = smooth_weight
        self.img_seg_loss_weights = img_seg_loss_weights
        self.clip_loss_weight = clip_loss_weight
        self.sample_rate = sample_rate
        self.img_seg_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.img_seg_loss_weights, 
                sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
        self.clip_loss = BridgePromptCLIPLoss(loss_weight=self.clip_loss_weight)
        self.elps = 1e-10

    def forward(self, model_output, input_data):
        # img_seg_score [stage_num N C T]
        # img_feature [N C T]
        # text_feature [N C T]
        # img_extract_score [N C T]
        text_all_embedding, text_cnt_embedding, text_acts_embedding, cnt_emb, image_embedding, img_seg_score = model_output

        # img backbone label learning
        img_seg_loss = self.img_seg_loss(img_seg_score, input_data)['loss']

        # all loss
        image_embedding_mean = image_embedding.mean(dim=1, keepdim=False)
        all_loss = self.clip_loss([image_embedding_mean, text_all_embedding], input_data)['loss']
        # cnt loss
        cnt_loss = self.clip_loss([cnt_emb, text_cnt_embedding], input_data)['loss']
        # act loss
        act_loss = 0.
        for dd in range(text_acts_embedding.shape[1]):
            act_loss += self.clip_loss([image_embedding[:, dd, :], text_acts_embedding[:, dd, :]], input_data)['loss']

        loss = img_seg_loss + all_loss + cnt_loss + act_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["img_extract_loss"] = all_loss + cnt_loss + act_loss
        loss_dict["img_seg_loss"] = img_seg_loss
        loss_dict["clip_loss"] = all_loss + cnt_loss + act_loss
        return loss_dict

@LOSSES.register()
class BridgePromptCLIPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, need_logit_scale=True) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.error_metric = nn.KLDivLoss(size_average=True, reduce=True)
        if need_logit_scale is True:
            self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        else:
            self.logit_scale = None
    
    def create_logits(self, x1, x2, logit_scale):
        x1 = x1 / x1.norm(dim=-1, keepdim=True)
        x2 = x2 / x2.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logits_per_x1 = logit_scale * x1 @ x2.t()
        logits_per_x2 = logit_scale * x2 @ x1.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_x1, logits_per_x2

    def klloss(self, prediction, label):
        batch_size = prediction.shape[0]
        probs1 = F.log_softmax(prediction, 1)
        probs2 = F.softmax(label * 10, 1)
        loss = self.error_metric(probs1, probs2) * batch_size
        return loss
    
    def gen_label(slef, labels):
        labels_idx_order = torch.unique_consecutive(labels, return_counts=False, return_inverse=False)
        labels_idx_order = list(labels_idx_order.detach().cpu().numpy())
        num = len(labels_idx_order)
        gt = np.zeros(shape=(num, num))
        for i, label in enumerate(labels_idx_order):
            for k in range(num):
                if labels_idx_order[k] == label:
                    gt[i, k] = 1
        return gt
    
    def forward(self, model_output, input_data):
        image_embedding, text_embedding = model_output
        masks, labels, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        logit_scale = self.logit_scale.exp()

        ground_truth = self.gen_label(labels)
        logits_per_image, logits_per_text = self.create_logits(image_embedding, text_embedding, logit_scale)
        loss_imgs = self.klloss(logits_per_image, ground_truth)
        loss_texts = self.klloss(logits_per_text, ground_truth)
        loss = (loss_imgs + loss_texts) / 2
        
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict