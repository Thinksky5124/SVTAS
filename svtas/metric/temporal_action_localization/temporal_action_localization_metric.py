'''
Author       : Thyssen Wen
Date         : 2022-12-12 16:19:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 21:13:56
Description  : file content
FilePath     : /SVTAS/svtas/metric/temporal_action_localization/temporal_action_localization_metric.py
'''
import os
import pandas as pd
import numpy as np
from ...utils.logger import get_logger
from ..base_metric import BaseMetric
from ..builder import METRIC
from ..temporal_action_segmentation.temporal_action_segmentation_metric_utils import get_labels_scores_start_end_time
from .utils import wrapper_compute_average_precision

@METRIC.register()
class TALocalizationMetric(BaseMetric):
    def __init__(self,
                 actions_map_file_path,
                 train_mode=False,
                 show_ovberlaps=[0.5, 0.75],
                 tiou_thresholds=np.linspace(0.5, 0.95, 10),
                 file_output=False,
                 score_output=False,
                 gt_file_need=True,
                 output_format="json",
                 output_dir="output/results/pred_gt_list/",
                 score_output_dir="output/results/analysis/"):
        super().__init__()
        self.file_output = file_output
        self.score_output = score_output
        self.gt_file_need = gt_file_need
        self.output_format = output_format
        self.output_dir = output_dir
        self.score_output_dir = score_output_dir
        self.train_mode = train_mode
        self.tiou_thresholds = tiou_thresholds
        self.show_ovberlaps = show_ovberlaps
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
            
        self.logger = get_logger("SVTAS")
        # localization score
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

        pred_detection = get_labels_scores_start_end_time(
            outputs_arr, recog_content, self.actions_dict)
        gt_detection = get_labels_scores_start_end_time(
            np.ones(outputs_arr.shape), gt_content, self.actions_dict)
        
        if self.file_output is True and self.train_mode is False:
            if self.gt_file_need is True:
                self._write_seg_file(gt_content, vid + '-gt', self.output_dir)
            self._write_seg_file(recog_content, vid + '-pred', self.output_dir)
            
        return [recog_content, gt_content, pred_detection, gt_detection]
        
    def _update_score(self, vid, pred_detection, gt_detection):
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
        
    def update(self, vid, ground_truth_batch, outputs):
        """update metrics during each iter
        """
        # list [N, T]
        predicted_batch = outputs['predict']
        # list [N, C, T]
        output_np_batch = outputs['output_np']

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
            self._update_score([vid[bs]], pred_detection, gt_detection)

            # count acc
            acc = 0.
            total = 1
            for p, t in zip(recog_content, gt_content):
                if p != t:
                    total += 1
                elif p == t:
                    total += 1
                    acc += 1
            single_batch_acc += acc

        return single_batch_acc / len(predicted_batch)
    
    def _compute_metrics(self):
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
        overlap = list(self.tiou_thresholds)
        for s in range(len(overlap)):
            metric_dict['mAP@{:0.2f}'.format(
                overlap[s])] = mAP[s]
        metric_dict['avg_mAP'] = average_mAP

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "Model performence in TAL task : "
        # localization metric
        for s in range(len(self.show_ovberlaps)):
            log_mertic_info += 'mAP@{:0.2f}: {:.4f}, '.format(
                self.show_ovberlaps[s],
                metric_dict['mAP@{:0.2f}'.format(self.show_ovberlaps[s])])
        log_mertic_info += "avg_mAP: {:.4f}, ".format(metric_dict['avg_mAP'])
        self.logger.info(log_mertic_info)

    def _clear_for_next_epoch(self):
        # clear for next epoch
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
        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict