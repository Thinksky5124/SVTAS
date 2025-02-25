'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 21:03:58
Description: model postprecessing
FilePath     : /ETESVS/model/post_precessings/stream_score_post_processing.py
'''
import numpy as np
import torch
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class StreamScorePostProcessing():
    def __init__(self,
                 num_classes,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 ignore_index=-100):
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
    
    def init_scores(self, sliding_num, batch_size):
        max_temporal_len = sliding_num * self.sliding_window + self.sample_rate * self.clip_seg_num
        sample_videos_max_len = max_temporal_len + \
            ((self.clip_seg_num * self.sample_rate) - max_temporal_len % (self.clip_seg_num * self.sample_rate))
        if sample_videos_max_len % self.sliding_window != 0:
            sample_videos_max_len = sample_videos_max_len + \
                (self.sliding_window - (sample_videos_max_len % self.sliding_window))
        self.sample_videos_max_len = sample_videos_max_len
        self.pred_scores = np.zeros((batch_size, self.num_classes, sample_videos_max_len))
        self.video_gt = np.full((batch_size, sample_videos_max_len), self.ignore_index)
        self.init_flag = True

    def update(self, seg_scores, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N C T]
        with torch.no_grad():
            start_frame = idx * self.sliding_window
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + (self.clip_seg_num * self.sample_rate)
            self.pred_scores[:, :, start_frame:end_frame] = seg_scores[-1, :].detach().cpu().numpy().copy()
            self.video_gt[:, start_frame:end_frame] = gt.detach().cpu().numpy().copy()
            pred = np.argmax(seg_scores[-1, :].detach().cpu().numpy(), axis=-2)
            acc = np.mean((np.sum(pred == gt.detach().cpu().numpy(), axis=1) / (np.sum(gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))
        return acc

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []

        for bs in range(self.pred_scores.shape[0]):
            index = np.where(self.video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [self.sample_videos_max_len])
            predicted = np.argmax(self.pred_scores[bs, :, :ignore_start], axis=0)
            predicted = predicted.squeeze()
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list

@POSTPRECESSING.register()
class StreamScorePostProcessingMultiLabel():
    def __init__(self,
                 num_action_classes,
                 num_branch_classes,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 ignore_index=-100):
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.num_action_classes = num_action_classes
        self.num_branch_classes = num_branch_classes
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10

    def init_scores(self, sliding_num, batch_size):
        max_temporal_len = sliding_num * self.sliding_window + self.sample_rate * self.clip_seg_num
        sample_videos_max_len = max_temporal_len + \
            ((self.clip_seg_num * self.sample_rate) - max_temporal_len % (self.clip_seg_num * self.sample_rate))
        if sample_videos_max_len % self.sliding_window != 0:
            sample_videos_max_len = sample_videos_max_len + \
                (self.sliding_window - (sample_videos_max_len % self.sliding_window))
        self.sample_videos_max_len = sample_videos_max_len

        # Initialize scores for action and branch outputs
        self.pred_action_scores = np.zeros((batch_size, self.num_action_classes, sample_videos_max_len))
        self.pred_branch_scores = np.zeros((batch_size, self.num_branch_classes, sample_videos_max_len))

        self.video_action_gt = np.full((batch_size, sample_videos_max_len), self.ignore_index)
        self.video_branch_gt = np.full((batch_size, sample_videos_max_len), self.ignore_index)

        self.init_flag = True

    def update(self, action_scores, branch_scores, action_gt, branch_gt, idx):
        # action_scores, branch_scores [stage_num N C T]
        # action_gt, branch_gt [N C T]
        with torch.no_grad():
            start_frame = idx * self.sliding_window
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + (self.clip_seg_num * self.sample_rate)

            # Update scores and ground truths for action
            self.pred_action_scores[:, :, start_frame:end_frame] = action_scores[-1, :].detach().cpu().numpy().copy()
            self.video_action_gt[:, start_frame:end_frame] = action_gt.detach().cpu().numpy().copy()

            # Update scores and ground truths for branch
            self.pred_branch_scores[:, :, start_frame:end_frame] = branch_scores[-1, :].detach().cpu().numpy().copy()
            self.video_branch_gt[:, start_frame:end_frame] = branch_gt.detach().cpu().numpy().copy()

            # Calculate accuracy for action and branch
            pred_action = np.argmax(action_scores[-1, :].detach().cpu().numpy(), axis=-2)
            acc_action = np.mean((np.sum(pred_action == action_gt.detach().cpu().numpy(), axis=1) / 
                                  (np.sum(action_gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))

            pred_branch = np.argmax(branch_scores[-1, :].detach().cpu().numpy(), axis=-2)
            acc_branch = np.mean((np.sum(pred_branch == branch_gt.detach().cpu().numpy(), axis=1) / 
                                  (np.sum(branch_gt.detach().cpu().numpy() != self.ignore_index, axis=1) + self.epls)))

        return acc_action, acc_branch

    def output(self):
        action_pred_score_list = []
        action_pred_cls_list = []
        action_ground_truth_list = []

        branch_pred_score_list = []
        branch_pred_cls_list = []
        branch_ground_truth_list = []

        for bs in range(self.pred_action_scores.shape[0]):
            # Action outputs
            action_index = np.where(self.video_action_gt[bs, :] == self.ignore_index)
            action_ignore_start = min(list(action_index[0]) + [self.sample_videos_max_len])
            action_predicted = np.argmax(self.pred_action_scores[bs, :, :action_ignore_start], axis=0)
            action_pred_cls_list.append(action_predicted.squeeze().copy())
            action_pred_score_list.append(self.pred_action_scores[bs, :, :action_ignore_start].copy())
            action_ground_truth_list.append(self.video_action_gt[bs, :action_ignore_start].copy())

            # Branch outputs
            branch_index = np.where(self.video_branch_gt[bs, :] == self.ignore_index)
            branch_ignore_start = min(list(branch_index[0]) + [self.sample_videos_max_len])
            branch_predicted = np.argmax(self.pred_branch_scores[bs, :, :branch_ignore_start], axis=0)
            branch_pred_cls_list.append(branch_predicted.squeeze().copy())
            branch_pred_score_list.append(self.pred_branch_scores[bs, :, :branch_ignore_start].copy())
            branch_ground_truth_list.append(self.video_branch_gt[bs, :branch_ignore_start].copy())

        return (action_pred_score_list, action_pred_cls_list, action_ground_truth_list), \
               (branch_pred_score_list, branch_pred_cls_list, branch_ground_truth_list)
