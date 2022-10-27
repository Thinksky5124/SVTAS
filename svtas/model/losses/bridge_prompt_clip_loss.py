'''
Author       : Thyssen Wen
Date         : 2022-06-15 19:43:47
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-29 10:10:56
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
                 cnt_max=7,
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
        self.cnt_max = cnt_max
        self.img_seg_loss = SegmentationLoss(num_classes=self.num_classes, loss_weight=self.img_seg_loss_weights, 
                sample_rate=self.sample_rate, smooth_weight=self.smooth_weight, ignore_index=self.ignore_index)
        self.clip_loss = BridgePromptCLIPLoss(loss_weight=self.clip_loss_weight, ignore_index=self.ignore_index)
        self.elps = 1e-10
    
    def gen_label(self, labels, dtype):
        labels_order_list = []
        labels_order_sum_list = []
        for batch_size in range(labels.shape[0]):
            labels_idx_order = torch.unique_consecutive(labels[batch_size, :], return_counts=False, return_inverse=False)

            # labels_idx_order [NUM] -> [cnt_max]
            if labels_idx_order.shape[0] >= self.cnt_max:
                labels_idx_order = labels_idx_order[:self.cnt_max]
            else:
                labels_idx_order = torch.cat([labels_idx_order,
                    torch.full([self.cnt_max], fill_value=self.ignore_index,
                    dtype=labels_idx_order.dtype,
                    device=labels_idx_order.device)[:(self.cnt_max - labels_idx_order.shape[0])]], dim=0)
            labels_idx_order_cnt = labels_idx_order >= 0
            labels_idx_order_cnt = torch.sum(labels_idx_order_cnt).unsqueeze(0)

            labels_order_list.append(labels_idx_order.unsqueeze(0))
            labels_order_sum_list.append(labels_idx_order_cnt.unsqueeze(0))
        
        # [N cnt_max]
        labels_idx_order = torch.concat(labels_order_list, dim=0)
        # [N 1]
        labels_idx_order_sum = torch.concat(labels_order_sum_list, dim=0)

        labels_idx_order_list = list(labels_idx_order.detach().cpu().numpy())
        labels_idx_order_sum = list(labels_idx_order_sum.detach().cpu().numpy())
        
        all_num = len(labels_idx_order_list)
        all_gt = np.zeros(shape=(all_num, all_num))
        for i, label in enumerate(labels_idx_order_list):
            for k in range(all_num):
                if np.all(np.equal(labels_idx_order_list[k], label)):
                    all_gt[i, k] = 1
        # [N N]
        all_ground_truth = torch.tensor(all_gt, dtype=dtype, device=labels.device)

        cnt_num = len(labels_idx_order_sum)
        cnt_gt = np.zeros(shape=(cnt_num, cnt_num))
        for i, label in enumerate(labels_idx_order_sum):
            for k in range(cnt_num):
                if labels_idx_order_sum[k] == label:
                    cnt_gt[i, k] = 1
        # [N N]
        cnt_ground_truth = torch.tensor(cnt_gt, dtype=dtype, device=labels.device)
        
        # act gt
        act_gt_list = []
        for act_pos_idx in range(labels_idx_order.shape[-1]):
            act_num = len(labels_idx_order[:, act_pos_idx])
            act_gt = np.zeros(shape=(act_num, act_num))
            for i, label in enumerate(labels_idx_order):
                for k in range(act_num):
                    if labels_idx_order[k][act_pos_idx] == label[act_pos_idx]:
                        act_gt[i, k] = 1
            act_gt = torch.tensor(act_gt, dtype=dtype, device=labels.device).unsqueeze(-1)
            act_gt_list.append(act_gt)
        # [N N cnt_max]
        act_ground_truth = torch.concat(act_gt_list, dim=-1)

        return all_ground_truth, cnt_ground_truth, act_ground_truth

    def forward(self, model_output, input_data):
        # img_seg_score [stage_num N C T]
        # image_embedding [N cnt_max D]
        # text_all_embedding [N D]
        # text_cnt_embedding [N D]
        # text_acts_embedding [N cnt_max D]
        # cnt_emb [N D]
        text_all_embedding, text_cnt_embedding, text_acts_embedding, cnt_emb, image_embedding, img_seg_score = model_output

        # img backbone label learning
        img_seg_loss = self.img_seg_loss(img_seg_score, input_data)['loss']

        all_ground_truth, cnt_ground_truth, act_ground_truth = self.gen_label(input_data["labels"], dtype=image_embedding.dtype)

        # generate bridge-prompt clip mask
        if torch.any(input_data["labels"] == self.ignore_index):
            bridge_prompt_mask = torch.zeros_like(all_ground_truth)
        else:
            bridge_prompt_mask = torch.ones_like(all_ground_truth)

        # all loss
        # [N NUM D] -> [N D]
        image_embedding_mean = torch.mean(image_embedding, dim=1)
        input_data_all = {"masks": bridge_prompt_mask, "labels": all_ground_truth, "precise_sliding_num": input_data["precise_sliding_num"]}
        all_loss = self.clip_loss([image_embedding_mean, text_all_embedding], input_data_all)['loss']
        # cnt loss
        input_data_cnt = {"masks": bridge_prompt_mask, "labels": cnt_ground_truth, "precise_sliding_num": input_data["precise_sliding_num"]}
        cnt_loss = self.clip_loss([cnt_emb, text_cnt_embedding], input_data_cnt)['loss']
        # act loss
        act_loss = 0.
        for dd in range(text_acts_embedding.shape[1]):
            input_data_act = {"masks": bridge_prompt_mask, "labels": act_ground_truth[:, :, dd], "precise_sliding_num": input_data["precise_sliding_num"]}
            act_loss += self.clip_loss([image_embedding[:, dd, :], text_acts_embedding[:, dd, :]], input_data_act)['loss']

        loss = img_seg_loss + all_loss + cnt_loss + act_loss

        loss_dict={}
        loss_dict["loss"] = loss
        loss_dict["img_extract_loss"] = all_loss + cnt_loss + act_loss
        loss_dict["img_seg_loss"] = img_seg_loss
        loss_dict["clip_loss"] = all_loss + cnt_loss + act_loss
        return loss_dict

@LOSSES.register()
class BridgePromptCLIPLoss(nn.Module):
    def __init__(self, loss_weight=1.0, need_logit_scale=True, ignore_index=-100) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight
        self.error_metric = nn.KLDivLoss(reduction="none")
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
    
    def forward(self, model_output, input_data):
        image_embedding, text_embedding = model_output
        masks, ground_truth, precise_sliding_num = input_data["masks"], input_data["labels"], input_data['precise_sliding_num']
        logit_scale = self.logit_scale.exp()

        # [N cnt_max cnt_max]
        logits_per_image, logits_per_text = self.create_logits(image_embedding, text_embedding, logit_scale)
        loss_imgs = self.klloss(logits_per_image, ground_truth)
        loss_texts = self.klloss(logits_per_text, ground_truth)
        loss = torch.mean(torch.mean((loss_imgs + loss_texts) / 2 * masks, dim=-1) / precise_sliding_num)
        
        loss_dict={}
        loss_dict["loss"] = loss * self.loss_weight
        return loss_dict