'''
Author       : Thyssen Wen
Date         : 2022-12-12 21:34:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-15 16:50:32
Description  : file content
FilePath     : /SVTAS/svtas/metric/svtas/svtas_metric.py
'''
import os
import numpy as np
from ..base_metric import BaseMetric
from ...utils import get_logger
from svtas.utils import AbstractBuildFactory

from ..tas.tas_metric_utils import get_labels_scores_start_end_time
from ..tas.tas_metric_utils import levenstein, edit_score, f_score

@AbstractBuildFactory.register('metric')
class SVTASegmentationMetric(BaseMetric):
    def __init__(self,
                 overlap,
                 actions_map_file_path,
                 segment_windows_size,
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
        self.segment_windows_size = segment_windows_size

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

        self.total_acc = 0.
        self.total_edit = 0.
        self.total_seg = 0
        self.total_f1 = [0. for _ in range(self.overlap_len)]

    def _update_score(self, recog_content, gt_content):
        current_f1 = [0. for _ in range(self.overlap_len)]
        current_acc = 0.
        current_seg_cnt = 0
        for start_frame in range(0, len(recog_content), self.segment_windows_size):
            self.total_seg += 1
            current_seg_cnt += 1
            # cls score
            correct = 0
            total = 0

            if start_frame + self.segment_windows_size > len(recog_content):
                end_frame = len(recog_content)
            else:
                end_frame = start_frame + self.segment_windows_size
            
            seg_gt_content = gt_content[start_frame:end_frame]
            seg_recog_content = recog_content[start_frame:end_frame]

            for i in range(len(seg_gt_content)):
                total += 1

                if seg_gt_content[i] == seg_recog_content[i]:
                    correct += 1

            acc = correct / total
            self.total_acc += acc
            current_acc += acc

            edit_num = edit_score(seg_recog_content, seg_gt_content)
            self.total_edit += edit_num

            for s in range(self.overlap_len):
                tp1, fp1, fn1 = f_score(seg_recog_content, seg_gt_content, self.overlap[s])

                # compute single f1
                precision = tp1 / float(tp1 + fp1 + self.elps)
                recall = tp1 / float(tp1 + fn1 + self.elps)

                f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)
                f1 = np.nan_to_num(f1)
                self.total_f1[s] += f1
                current_f1[s] += f1

        return current_f1[-1] / current_seg_cnt, current_acc / current_seg_cnt

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

    def update(self, vid, ground_truth_batch, outputs):
        """update metrics during each iter
        """
        # list [N, T]
        predicted_batch = outputs['predict']
        # list [N, C, T]
        output_np_batch = outputs['output_np']

        single_batch_f1 = 0.
        single_batch_acc = 0.

        if len(vid) != len(predicted_batch):
            repet_rate = len(predicted_batch) // len(vid)
            vid = [val for val in vid for i in range(repet_rate)]
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
            
            if self.score_output is True and self.train_mode is False:
                score_output_path = os.path.join(self.score_output_dir, vid[bs] + ".npy")
                np.save(score_output_path, output_np)

            result = self._transform_model_result(vid[bs], outputs_np, gt_np, outputs_arr)
            recog_content, gt_content, pred_detection, gt_detection = result
            single_f1, acc = self._update_score(recog_content, gt_content)
            single_batch_f1 += single_f1
            single_batch_acc += acc
        return single_batch_acc / len(predicted_batch)

    def _compute_metrics(self):
        # cls metric
        Acc = 100 * float(self.total_acc) / (self.total_seg + self.elps)
        Edit = (1.0 * self.total_edit) / (self.total_seg + self.elps)
        Fscore = dict()
        for s in range(self.overlap_len):
            f1 = self.total_f1[s] / self.total_seg
            Fscore[self.overlap[s]] = f1 * 100

        # save metric
        metric_dict = dict()
        metric_dict['mAcc'] = Acc
        metric_dict['mEdit'] = Edit
        for s in range(len(self.overlap)):
            metric_dict['mF1@{:0.2f}'.format(
                self.overlap[s])] = Fscore[self.overlap[s]]

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "Model performence in SVTAS task : "
        # preds ensemble
        log_mertic_info += "mAcc: {:.4f}, ".format(metric_dict['mAcc'])
        log_mertic_info += 'mEdit: {:.4f}, '.format(metric_dict['mEdit'])
        for s in range(len(self.overlap)):
            log_mertic_info += 'mF1@{:0.2f}: {:.4f}, '.format(
                self.overlap[s],
                metric_dict['mF1@{:0.2f}'.format(self.overlap[s])])

        self.logger.info(log_mertic_info)

    def _clear_for_next_epoch(self):
        # clear for next epoch
        # cls
        self.total_acc = 0.
        self.total_edit = 0.
        self.total_seg = 0
        self.total_f1 = [0. for _ in range(self.overlap_len)]

    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict