import numpy as np
import argparse
import pandas as pd

from utils.config import get_logger
import os

from .metric_utils import get_labels_scores_start_end_time, get_labels_start_end_time
from .metric_utils import levenstein, edit_score, f_score, boundary_AR
from .metric_utils import wrapper_compute_average_precision

class BaseSegmentationMetric(object):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 overlap,
                 actions_map_file_path,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 file_output=False,
                 output_dir="output/results/pred_gt_list/"):
        """prepare for metrics
        """
        self.logger = get_logger("ETETS")
        self.elps = 1e-10
        self.file_output = file_output
        self.output_dir = output_dir

        if self.file_output is True:
            isExists = os.path.exists(self.output_dir)
            if not isExists:
                os.makedirs(self.output_dir)

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
        return f1

    def _write_seg_file(self, input_data, write_name, write_path):
        recog_content = [line + "\n" for line in input_data]

        write_path = os.path.join(write_path, write_name + ".txt")
        f = open(write_path, "w")
        f.writelines(recog_content)
        f.close()

    def _transform_model_result(self, vid, outputs_np, gt_np, outputs_arr):
        recognition = []
        for i in range(outputs_np.shape[0]):
            recognition = np.concatenate((recognition, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(outputs_np[i])]
            ]))
        recog_content = list(recognition)
       
        gt_content = []
        for i in range(gt_np.shape[0]):
            gt_content = np.concatenate((gt_content, [
                list(self.actions_dict.keys())[list(
                    self.actions_dict.values()).index(gt_np[i])]
            ]))
        gt_content = list(gt_content)

        if self.file_output is True:
            self._write_seg_file(gt_content, vid + '-gt', self.output_dir)
            self._write_seg_file(recog_content, vid + '-pred', self.output_dir)

        pred_detection = get_labels_scores_start_end_time(
            outputs_arr, recog_content, self.actions_dict)
        gt_detection = get_labels_scores_start_end_time(
            np.ones(outputs_arr.shape), gt_content, self.actions_dict)

        return [recog_content, gt_content, pred_detection, gt_detection]

    def update(self, outputs):
        """update metrics during each iter
        """
        raise NotImplementedError

    def _compute_metrics(self):
        # cls metric
        Acc = 100 * float(self.total_correct) / self.total_frame
        Edit = (1.0 * self.total_edit) / self.total_video
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

class SegmentationMetric(BaseSegmentationMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 overlap,
                 actions_map_file_path,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 file_output=False,
                 output_dir="output/results/pred_gt_list/"):
        """prepare for metrics
        """
        super().__init__(overlap, actions_map_file_path,
                         max_proposal, tiou_thresholds,
                         file_output, output_dir)
    
    def update(self, vid, ground_truth_batch, outputs):
        """update metrics during each iter
        """
        # list [N, T]
        predicted_batch = outputs['predict']
        # list [N, C, T]
        output_np_batch = outputs['output_np']

        single_batch_f1 = 0.
        for bs in range(len(predicted_batch)):
            predicted = predicted_batch[bs]
            output_np = output_np_batch[bs]
            groundTruth = ground_truth_batch[bs]

            if type(predicted) is not np.ndarray:
                outputs_np = predicted.numpy()
                outputs_arr = output_np.numpy()
                gt_np = groundTruth.numpy()
            else:
                outputs_np = predicted
                outputs_arr = output_np
                gt_np = groundTruth

            result = self._transform_model_result(vid[bs], outputs_np, gt_np, outputs_arr)
            recog_content, gt_content, pred_detection, gt_detection = result
            single_f1 = self._update_score([vid[bs]], recog_content, gt_content, pred_detection,
                            gt_detection)
            single_batch_f1 += single_f1
        
        return single_batch_f1 / len(predicted_batch)

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict
