'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:11:13
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 22:46:28
Description  : Temporal action segmentation base class
FilePath     : /SVTAS/svtas/metric/temporal_action_segmentation/temporal_action_segmentation_base_class.py
'''
import os
import numpy as np
from ..base_metric import BaseMetric
from ...utils.config import get_logger
from ..builder import METRIC

from .tas_metric_utils import get_labels_scores_start_end_time
from .tas_metric_utils import levenstein, edit_score, f_score

@METRIC.register()
class BaseTASegmentationMetric(BaseMetric):
    """
    Test for Video Segmentation based model.
    """

    def __init__(self,
                 overlap,
                 actions_map_file_path,
                 train_mode=False,
                 file_output=False,
                 score_output=False,
                 gt_file_need=True,
                 output_format="txt",
                 output_dir="output/results/pred_gt_list/",
                 score_output_dir="output/results/analysis/"):
        """prepare for metrics
        """
        super().__init__()
        self.logger = get_logger("SVTAS")
        self.elps = 1e-10
        self.file_output = file_output
        self.score_output = score_output
        self.gt_file_need = gt_file_need
        self.output_format = output_format
        self.output_dir = output_dir
        self.score_output_dir = score_output_dir
        self.train_mode = train_mode

        assert self.output_format in ["txt", "json"], "Unsupport output format: " + self.output_format + "!"

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

    def _update_score(self, recog_content, gt_content):
        # cls score
        correct = 0
        total = 0

        for i in range(len(gt_content)):
            total += 1
            #accumulate
            self.total_frame += 1

            if gt_content[i] == recog_content[i]:
                correct += 1
                #accumulate
                self.total_correct += 1

        edit_num = edit_score(recog_content, gt_content)
        self.total_edit += edit_num

        for s in range(self.overlap_len):
            tp1, fp1, fn1 = f_score(recog_content, gt_content, self.overlap[s])

            # accumulate
            self.cls_tp[s] += tp1
            self.cls_fp[s] += fp1
            self.cls_fn[s] += fn1

        # accumulate
        self.total_video += 1

        # compute single f1
        precision = tp1 / float(tp1 + fp1 + self.elps)
        recall = tp1 / float(tp1 + fn1 + self.elps)

        f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

        f1 = np.nan_to_num(f1)
        acc = correct / total
        return f1, acc

    def _write_seg_file(self, input_data, write_name, write_path):
        if self.output_format in ['txt']:
            self._write_txt_format(input_data, write_name, write_path)
        elif self.output_format in ['json']:
            self._write_json_format(input_data, write_name, write_path)
        else:
            raise NotImplementedError

    def _write_json_format(self, input_data, write_name, write_path):
        pass

    def _write_txt_format(self, input_data, write_name, write_path):
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

        if self.file_output is True and self.train_mode is False:
            if self.gt_file_need is True:
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
        Acc = 100 * float(self.total_correct) / (self.total_frame + self.elps)
        Edit = (1.0 * self.total_edit) / (self.total_video + self.elps)
        Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fp[s] + self.elps)
            recall = self.cls_tp[s] / float(self.cls_tp[s] + self.cls_fn[s] + self.elps)

            f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

            f1 = np.nan_to_num(f1) * 100
            Fscore[self.overlap[s]] = f1

        # save metric
        metric_dict = dict()
        metric_dict['Acc'] = Acc
        metric_dict['Edit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['F1@{:0.2f}'.format(
                self.overlap[s])] = Fscore[self.overlap[s]]

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "Model performence in TAS task : "
        # preds ensemble
        log_mertic_info += "Acc: {:.4f}, ".format(metric_dict['Acc'])
        log_mertic_info += 'Edit: {:.4f}, '.format(metric_dict['Edit'])
        for s in range(len(self.overlap)):
            log_mertic_info += 'F1@{:0.2f}: {:.4f}, '.format(
                self.overlap[s],
                metric_dict['F1@{:0.2f}'.format(self.overlap[s])])

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

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        raise NotImplementedError