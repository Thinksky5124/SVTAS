import numpy as np
import torch

class PostProcessing(object):
    def __init__(self,
                 num_classes,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 clip_buffer_num):
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.clip_buffer_num = clip_buffer_num
        self.num_classes = num_classes
        self.init_flag = False

    def init_scores(self, sliding_num, batch_size):
        max_temporal_len = sliding_num * self.sliding_window + self.sample_rate * self.clip_seg_num
        sample_videos_max_len = max_temporal_len + \
            ((self.clip_seg_num * self.sample_rate) - max_temporal_len % (self.clip_seg_num * self.sample_rate))
        if sample_videos_max_len % self.sliding_window != 0:
            sample_videos_max_len = sample_videos_max_len + \
                (self.sliding_window - (sample_videos_max_len % self.sliding_window))
        self.sample_videos_max_len = sample_videos_max_len
        self.pred_scores = torch.zeros((batch_size, self.num_classes, sample_videos_max_len))
        self.video_gt = torch.zeros((batch_size, sample_videos_max_len))
        self.init_flag = True
    
    def update(self, seg_scores, gt, idx):
        with torch.no_grad():
            start_frame = idx * self.sliding_window - (self.clip_buffer_num * self.clip_seg_num) * self.sample_rate
            if start_frame < 0:
                start_frame = 0
            end_frame = start_frame + (self.clip_seg_num * self.sample_rate)
            self.pred_scores[:, :, start_frame:end_frame] = seg_scores[-1, :]
            self.video_gt[:, start_frame:end_frame] = gt

    def output(self):
        pred_score_list = []
        pred_cls_list = []
        ground_truth_list = []

        for bs in range(self.pred_scores.shape[0]):
            index = np.where(self.video_gt[bs, :].cpu().numpy() == -100)
            ignore_start = min(list(index[0]) + [self.sample_videos_max_len])
            predicted = torch.argmax(self.pred_scores[bs, :, :ignore_start], axis=0)
            predicted = predicted.squeeze().cpu().numpy()
            pred_cls_list.append(predicted)
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].cpu().numpy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].cpu().numpy())

        return pred_score_list, pred_cls_list, ground_truth_list