'''
Author       : Thyssen Wen
Date         : 2022-12-12 16:22:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-16 13:28:49
Description  : file content
FilePath     : /SVTAS/svtas/metric/tap/tap_metric.py
'''
import os
import numpy as np
from svtas.utils.logger import get_logger
from ..base_metric import BaseMetric
from svtas.utils import AbstractBuildFactory
from ..tas.tas_metric_utils import get_labels_scores_start_end_time
from .utils import boundary_AR

@AbstractBuildFactory.register('metric')
class TAProposalMetric(BaseMetric):
    def __init__(self,
                 actions_map_file_path,
                 train_mode=False,
                 max_proposal=100,
                 tiou_thresholds=np.linspace(0.1, 0.5, 4),
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
        # boundary score
        self.max_proposal = max_proposal
        self.AR_at_AN = [[] for _ in range(max_proposal)]
    
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
    
    def _update_score(self, pred_detection, gt_detection):
        # proposal score
        for AN in range(self.max_proposal):
            AR = boundary_AR(pred_detection,
                             gt_detection,
                             list(self.tiou_thresholds),
                             max_proposal=(AN + 1))
            self.AR_at_AN[AN].append(AR)

        return AR
    
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
            ar = self._update_score(pred_detection, gt_detection)
            
            # count acc
            acc = 0.
            total = 1
            for p, t in zip(recog_content, gt_content):
                if p != t:
                    total += 1
                elif p == t:
                    total += 1
                    acc += 1
            acc = acc / total
            single_batch_acc += acc
        return single_batch_acc / len(predicted_batch)
    
    def _compute_metrics(self):
        # proposal metric
        proposal_AUC = np.array(self.AR_at_AN) * 100
        AUC = np.mean(proposal_AUC)
        AR_at_AN1 = np.mean(proposal_AUC[0, :])
        AR_at_AN5 = np.mean(proposal_AUC[4, :])
        AR_at_AN15 = np.mean(proposal_AUC[14, :])

        # save metric
        metric_dict = dict()
        metric_dict['Auc'] = AUC
        metric_dict['AR@AN1'] = AR_at_AN1
        metric_dict['AR@AN5'] = AR_at_AN5
        metric_dict['AR@AN15'] = AR_at_AN15

        return metric_dict

    def _log_metrics(self, metric_dict):
        # log metric
        log_mertic_info = "Model performence in TAP task : "

        # boundary metric
        log_mertic_info += "Auc: {:.4f}, ".format(metric_dict['Auc'])
        log_mertic_info += "AR@AN1: {:.4f}, ".format(metric_dict['AR@AN1'])
        log_mertic_info += "AR@AN5: {:.4f}, ".format(metric_dict['AR@AN5'])
        log_mertic_info += "AR@AN15: {:.4f}, ".format(metric_dict['AR@AN15'])

        self.logger.info(log_mertic_info)

    def _clear_for_next_epoch(self):
        # clear for next epoch
        # proposal
        self.AR_at_AN = [[] for _ in range(self.max_proposal)]
    
    def accumulate(self):
        """accumulate metrics when finished all iters.
        """
        metric_dict = self._compute_metrics()
        self._log_metrics(metric_dict)
        self._clear_for_next_epoch()

        return metric_dict