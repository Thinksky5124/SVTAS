'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:11:13
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 10:01:05
Description  : Temporal action segmentation base class
FilePath     : /ETESVS/metric/temporal_action_segmentation/temporal_action_segmentation_base_class.py
'''
import os
import numpy as np
import pandas as pd
from ..base_metric import BaseMetric
from utils.config import get_logger
from ..builder import METRIC

from .temporal_action_segmentation_metric_utils import get_labels_scores_start_end_time, get_labels_start_end_time
from .temporal_action_segmentation_metric_utils import levenstein, edit_score, f_score, boundary_AR
from .temporal_action_segmentation_metric_utils import wrapper_compute_average_precision

@METRIC.register()
class BaseTASegmentationMetric(BaseMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 overlap,
                 actions_map_file_path,
                 train_mode=False,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 file_output=False,
                 score_output=False,
                 output_dir="output/results/pred_gt_list/",
                 score_output_dir="output/results/analysis/"):
        """prepare for metrics
        """
        super().__init__()
        self.logger = get_logger("SVTAS")
        self.elps = 1e-10
        self.file_output = file_output
        self.score_output = score_output
        self.output_dir = output_dir
        self.score_output_dir = score_output_dir
        self.train_mode = train_mode

        if self.file_output is True and self.train_mode is False:
            isExists = os.path.exists(self.output_dir)
            if not isExists:
                os.makedirs(self.output_dir)

        if self.score_output is True and self.train_mode is False:
            isExists = os.path.exists(self.score_output_dir)
            if not isExists:
                os.makedirs(self.score_output_dir)

        # actions dict generate
        file_ptr = open(actions_map_file_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[a.split()[1]] = int(a.split()[0])
        # cls score
        self.overlap = overlap
        self.overlap_len = len(overlap)

        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0

        # boundary score
        self.max_proposal = max_proposal
        self.AR_at_AN = [[] for _ in range(max_proposal)]

        # localization score
        self.tiou_thresholds = tiou_thresholds
        self.pred_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": [],
            "score": []
        }
        self.gt_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": []
        }

    def _update_score(self, vid, recog_content, gt_content, pred_detection,
                      gt_detection):
        # cls score
        correct = 0
        total = 0
        edit = 0

        for i in range(len(gt_content)):
            total += 1
            #accumulate
            self.total_frame += 1

            if gt_content[i] == recog_content[i]:
                correct += 1
                #accumulate
                self.total_correct += 1

        edit_num = edit_score(recog_content, gt_content)
        edit += edit_num
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, self.overlap[s])

            # accumulate
            self.cls_tp[s] += tp1
            self.cls_fp[s] += fp1
            self.cls_fn[s] += fn1

        # accumulate
        self.total_video += 1

        # proposal score
        for AN in range(self.max_proposal):
            AR = boundary_AR(pred_detection,
                             gt_detection,
                             self.overlap,
                             max_proposal=(AN + 1))
            self.AR_at_AN[AN].append(AR)

        # localization score

        p_label, p_start, p_end, p_scores = pred_detection
        g_label, g_start, g_end, _ = gt_detection
        p_vid_list = vid * len(p_label)
        g_vid_list = vid * len(g_label)

        # collect
        self.pred_results_dict[
            "video-id"] = self.pred_results_dict["video-id"] + p_vid_list
        self.pred_results_dict[
            "t_start"] = self.pred_results_dict["t_start"] + p_start
        self.pred_results_dict[
            "t_end"] = self.pred_results_dict["t_end"] + p_end
        self.pred_results_dict[
            "label"] = self.pred_results_dict["label"] + p_label
        self.pred_results_dict[
            "score"] = self.pred_results_dict["score"] + p_scores

        self.gt_results_dict[
            "video-id"] = self.gt_results_dict["video-id"] + g_vid_list
        self.gt_results_dict[
            "t_start"] = self.gt_results_dict["t_start"] + g_start
        self.gt_results_dict["t_end"] = self.gt_results_dict["t_end"] + g_end
        self.gt_results_dict["label"] = self.gt_results_dict["label"] + g_label

        # compute single f1
        precision = tp1 / float(tp1 + fp1 + self.elps)
        recall = tp1 / float(tp1 + fn1 + self.elps)

        f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

        f1 = np.nan_to_num(f1)
        acc = correct / total
        return f1, acc

    def _write_seg_file(self, input_data, write_name, write_path):
        recog_content = [line + "\n" for line in input_data]

        write_path = os.path.join(write_path, write_name + ".txt")
        f = open(write_path, "w")
        f.writelines(recog_content)
        f.close()

    def _transform_model_result(self, vid, outputs_np, gt_np, outputs_arr, action_dict=None):
        recognition = []

        if action_dict is not None: 
            act_dict = action_dict
        else:
            act_dict = self.actions_dict

        for i in range(outputs_np.shape[0]):
            recognition = np.concatenate((recognition, [
                list(act_dict.keys())[list(
                    act_dict.values()).index(outputs_np[i])]
            ]))
        recog_content = list(recognition)
       
        gt_content = []
        for i in range(gt_np.shape[0]):
            gt_content = np.concatenate((gt_content, [
                list(act_dict.keys())[list(
                    act_dict.values()).index(gt_np[i])]
            ]))
        gt_content = list(gt_content)

        if self.file_output is True and self.train_mode is False:
            self._write_seg_file(gt_content, vid + f'-{list(act_dict.keys())[0]}-gt', self.output_dir)
            self._write_seg_file(recog_content, vid + f'-{list(act_dict.keys())[0]}-pred', self.output_dir)
        try:
            pred_detection = get_labels_scores_start_end_time(
                outputs_arr, recog_content, act_dict)
            gt_detection = get_labels_scores_start_end_time(
                np.ones(outputs_arr.shape), gt_content, act_dict)
        except:
            import pdb; pdb.set_trace()
        return [recog_content, gt_content, pred_detection, gt_detection]

    def update(self, outputs):
        """update metrics during each iter
        """
        raise NotImplementedError

    def _compute_metrics(self):
        # cls metric
        Acc = 100 * float(self.total_correct) / (self.total_frame + self.elps)
        Edit = (1.0 * self.total_edit) / (self.total_video + self.elps)
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fp[s] + self.elps)
            recall = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fn[s] + self.elps)

            f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # proposal metric
        proposal_AUC = np.array(self.AR_at_AN) * 100
        AUC = np.mean(proposal_AUC)
        AR_at_AN1 = np.mean(proposal_AUC[0, :])
        AR_at_AN5 = np.mean(proposal_AUC[4, :])
        AR_at_AN15 = np.mean(proposal_AUC[14, :])

        # localization metric
        prediction = pd.DataFrame(self.pred_results_dict)
        ground_truth = pd.DataFrame(self.gt_results_dict)

        ap = wrapper_compute_average_precision(prediction, ground_truth,
                                               self.tiou_thresholds,
                                               self.actions_dict)

        mAP = ap.mean(axis=1) * 100
        average_mAP = mAP.mean()

        # save metric
        metric_dict = dict()
        metric_dict['Acc'] = Acc
        metric_dict['Edit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['F1@{:0.2f}'.format(
                self.overlap[s])] = Fscore[self.overlap[s]]
        metric_dict['Auc'] = AUC
        metric_dict['AR@AN1'] = AR_at_AN1
        metric_dict['AR@AN5'] = AR_at_AN5
        metric_dict['AR@AN15'] = AR_at_AN15
        metric_dict['mAP@0.5'] = mAP[0]
        metric_dict['avg_mAP'] = average_mAP

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "dataset model performence: "
        # preds ensemble
        log_mertic_info += "Acc: {:.4f}, ".format(metric_dict['Acc'])
        log_mertic_info += 'Edit: {:.4f}, '.format(metric_dict['Edit'])
        for s in range(len(self.overlap)):
            log_mertic_info += 'F1@{:0.2f}: {:.4f}, '.format(
                self.overlap[s],
                metric_dict['F1@{:0.2f}'.format(self.overlap[s])])

        # boundary metric
        log_mertic_info += "Auc: {:.4f}, ".format(metric_dict['Auc'])
        log_mertic_info += "AR@AN1: {:.4f}, ".format(metric_dict['AR@AN1'])
        log_mertic_info += "AR@AN5: {:.4f}, ".format(metric_dict['AR@AN5'])
        log_mertic_info += "AR@AN15: {:.4f}, ".format(metric_dict['AR@AN15'])

        # localization metric
        log_mertic_info += "mAP@0.5: {:.4f}, ".format(metric_dict['mAP@0.5'])
        log_mertic_info += "avg_mAP: {:.4f}, ".format(metric_dict['avg_mAP'])
        self.logger.info(log_mertic_info)

    def _clear_for_next_epoch(self):
        # clear for next epoch
        # cls
        self.cls_tp = np.zeros(self.overlap_len)
        self.cls_fp = np.zeros(self.overlap_len)
        self.cls_fn = np.zeros(self.overlap_len)
        self.total_correct = 0
        self.total_edit = 0
        self.total_frame = 0
        self.total_video = 0
        # proposal
        self.AR_at_AN = [[] for _ in range(self.max_proposal)]
        # localization
        self.pred_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": [],
            "score": []
        }
        self.gt_results_dict = {
            "video-id": [],
            "t_start": [],
            "t_end": [],
            "label": []
        }

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        raise NotImplementedError