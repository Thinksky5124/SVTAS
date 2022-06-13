'''
Author       : Thyssen Wen
Date         : 2022-06-13 16:56:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-13 17:15:01
Description  : Local Burr Suppression ref:https://github.com/lyhisme/ETSN
FilePath     : /ETESVS/model/post_precessings/lbs.py
'''
import torch
import numpy as np
from ..builder import POSTPRECESSING

@POSTPRECESSING.register()
class StreamScorePostProcessingWithLBS():
    def __init__(self,
                 num_classes,
                 clip_seg_num,
                 sliding_window,
                 sample_rate,
                 actions_map_file_path,
                 lbs_burr=7,
                 lbs_window=40,
                 lbs_Confidence=3.1,
                 ignore_index=-100,
                 bg_class_name=["background", "None"]):
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.sample_rate = sample_rate
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.init_flag = False
        self.epls = 1e-10
        self.bg_class = bg_class_name
        self.lbs_window = lbs_window
        self.lbs_Confidence = lbs_Confidence
        self.lbs_burr = lbs_burr

        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.id2class_map = dict()
        for a in actions:
            self.id2class_map[int(a.split()[0])] = a.split()[1]
    
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
            predicted = self.lbs(pred=predicted, pred_score=self.pred_scores[bs, :, :ignore_start])
            pred_cls_list.append(predicted.copy())
            pred_score_list.append(self.pred_scores[bs, :, :ignore_start].copy())
            ground_truth_list.append(self.video_gt[bs, :ignore_start].copy())

        return pred_score_list, pred_cls_list, ground_truth_list
    
    def get_segments(self, frame_wise_label):
        """
            Args:
                frame-wise label: frame-wise prediction or ground truth. 1D numpy array
            Return:
                segment-label array: list (excluding background class)
                start index list
                end index list
        """
        

        labels = []
        starts = []
        ends = []

        frame_wise_label = [
            self.id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))]

        # get class, start index and end index of segments
        # background class is excluded
        last_label = frame_wise_label[0]
        if frame_wise_label[0] not in self.bg_class:
            labels.append(frame_wise_label[0])
            starts.append(0)

        for i in range(len(frame_wise_label)):
            # if action labels change
            if frame_wise_label[i] not in last_label:
                # if label change from one class to another class
                # it's an action starting point
                if frame_wise_label[i] not in self.bg_class:
                    labels.append(frame_wise_label[i])
                    starts.append(i)

                # if label change from background to a class
                # it's not an action end point.
                if last_label not in self.bg_class:
                    ends.append(i)

                # update last label
                last_label = frame_wise_label[i]

        if last_label not in self.bg_class:
            ends.append(i)

        return labels, starts, ends

    def lbs(self, pred, pred_score):
        # remove short burrs
        p_label, p_start, p_end = self.get_segments(pred)
        for i in range(len(p_label)):
            if ((p_end[i] - p_start[i]) <= self.lbs_burr):
                if(i == 0):
                    for k in range(p_end[i], p_start[i]-1, -1):
                        pred[k] = pred[k+1]               
                else: 
                    for k in range(p_start[i], p_end[i]+1):
                        pred[k] = pred[k-1]

        confidence_burr = []
        p_label, p_start, p_end = self.get_segments(pred)
        # locate the burrs
        for i in range(len(p_label)):
            if ((p_end[i] - p_start[i]) <= self.lbs_window): 
                sum =[]
                for m in range(p_start[i],p_end[i]):
                    sum.append(np.max(pred_score[:,m:m+1]))
                if(np.mean(sum) <= self.lbs_Confidence):
                    confidence_burr.append(1)
                else:
                    confidence_burr.append(0)
            else:
                confidence_burr.append(0)

        count, old_index, consecutive_index= 0, 0, 0
        while (count < len(confidence_burr)):
            if (confidence_burr[count]):
                consecutive_index = 0
                old_index = count
                while confidence_burr[count]:
                    consecutive_index += 1
                    count += 1
                    if (count >= len(confidence_burr)):
                        break
                for j in range (consecutive_index):
                    confidence_burr[old_index+j] = consecutive_index
            else:
                count += 1

        index = 0
        # remove consecutive burrs
        while (index < len(confidence_burr)):
            if(confidence_burr[index] > 1):
                if(index + confidence_burr[index]-1 == len(confidence_burr)-1):
                    for k in range(p_start[index], p_end[index + int(confidence_burr[index])-1]+1):
                        pred[k] = pred[p_start[index]-1]
                elif(index + confidence_burr[index]-1 == 0):
                    for k in range(p_end[index + int(confidence_burr[index])-1], p_start[index]-1, -1):
                        pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
                else:
                    if (confidence_burr[index] % 2 == 0):
                        for k in range(p_start[index], p_end[index + int(confidence_burr[index]/2)-1]+1):
                            pred[k] = pred[p_start[index]-1]
                        for k in range(p_end[index + int(confidence_burr[index])-1], p_end[index + int(confidence_burr[index]/2)-1]-1, -1):
                            pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
                    else:
                        for k in range(p_start[index], p_end[index + int(confidence_burr[index]/2)-1]+1):
                            pred[k] = pred[p_start[index]-1]
                        for k in range(p_end[index + int(confidence_burr[index])-1], p_end[index + int(confidence_burr[index]/2)]-1, -1):
                            pred[k] = pred[p_end[index + int(confidence_burr[index])-1]+1]
                index += confidence_burr[index] -1
            index += 1

        confidence_burr = []
        # locate the burrs
        p_label, p_start, p_end = self.get_segments(pred)
        for i in range(len(p_label)):
            if ((p_end[i] - p_start[i]) <= self.lbs_window): 
                sum =[]
                for m in range(p_start[i],p_end[i]):
                    sum.append(np.max(pred_score[:,m:m+1]))
                if(np.mean(sum) <= self.lbs_Confidence):
                    confidence_burr.append(1)
                else:
                    confidence_burr.append(0)
            else:
                confidence_burr.append(0)
        # remove isolated burrs
        for i in range(len(p_label)):
            if(confidence_burr[i] == 1):
                if(i == 0):
                    for k in range(p_end[i], p_start[i]-1, -1):
                        pred[k] = pred[k+1]               
                elif(i == (len(confidence_burr)-1)): 
                    for k in range(p_start[i], p_end[i]+1):
                        pred[k] = pred[k-1]
                else:
                    q = int((p_end[i-1]-p_start[i-1])/(p_end[i-1]-p_start[i-1]+p_end[i+1]-p_start[i+1])*(p_end[i]-p_start[i]))
                    for k in range(p_start[i], p_start[i]+q+1):
                        pred[k] = pred[p_start[i]-1]
                    for k in range(p_end[i], p_start[i]+q-1, -1):
                        pred[k] = pred[p_end[i]+1]

        return pred