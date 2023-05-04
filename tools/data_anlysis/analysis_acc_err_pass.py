'''
Author       : Thyssen Wen
Date         : 2023-04-24 22:56:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-26 10:31:52
Description  : file content
FilePath     : /SVTAS/tools/data_anlysis/analysis_acc_err_pass.py
'''
import argparse
import os
from tqdm import tqdm
from prettytable import PrettyTable

import numpy as np

ignore_bg_class = ["background", "None"]

def get_labels_start_end_time(frame_wise_labels, bg_class=ignore_bg_class):
    labels = []
    starts = []
    ends = []
    last_label = frame_wise_labels[0]
    if frame_wise_labels[0] not in bg_class:
        labels.append(frame_wise_labels[0])
        starts.append(0)
    for i in range(len(frame_wise_labels)):
        if frame_wise_labels[i] != last_label:
            if frame_wise_labels[i] not in bg_class:
                labels.append(frame_wise_labels[i])
                starts.append(i)
            if last_label not in bg_class:
                ends.append(i)
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
    return labels, starts, ends

class SegErrorPassAnalysis(object):
    def __init__(self,
                 overlap=[.1, .25, .5, .75, .9]) -> None:
        self.overlap = overlap
        self.overlap_len = len(overlap)
        self.before_cls_tp = np.zeros(self.overlap_len)
        self.before_cls_fp = np.zeros(self.overlap_len)
        self.before_cls_fn = np.zeros(self.overlap_len)
        self.curr_cls_tp = np.zeros(self.overlap_len)
        self.curr_cls_fp = np.zeros(self.overlap_len)
        self.curr_cls_fn = np.zeros(self.overlap_len)
        self.errpr_pass_count_dict = [{'tt':0, 'tf':0, 'ff':0, 'ft':0} for _ in range(self.overlap_len - 1)]
        self.table = [PrettyTable() for _ in range(self.overlap_len - 1)]
        for i in range(self.overlap_len - 1):
            self.table[i].field_names = ["model_idx", f"Metric_{self.overlap[i]}", f"Metric_{self.overlap[i + 1]}","Error_pass_rate", "correction_capability"]

        self.elps = 1e-10
    
    def reset(self):
        self.before_cls_tp = np.zeros(self.overlap_len)
        self.before_cls_fp = np.zeros(self.overlap_len)
        self.before_cls_fn = np.zeros(self.overlap_len)
        self.curr_cls_tp = np.zeros(self.overlap_len)
        self.curr_cls_fp = np.zeros(self.overlap_len)
        self.curr_cls_fn = np.zeros(self.overlap_len)
    
    def f_score(self, recognized, ground_truth, overlap, bg_class=ignore_bg_class):
        p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

        tp = 0
        fp = 0
        if len(y_label) > 0:
            hits = np.zeros(len(y_label))

            for j in range(len(p_label)):
                intersection = np.minimum(p_end[j], y_end) - np.maximum(
                    p_start[j], y_start)
                union = np.maximum(p_end[j], y_end) - np.minimum(
                    p_start[j], y_start)
                IoU = (1.0 * intersection / union) * (
                    [p_label[j] == y_label[x] for x in range(len(y_label))])
                # Get the best scoring segment
                idx = np.array(IoU).argmax()

                if IoU[idx] >= overlap and not hits[idx]:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1
            fn = len(y_label) - sum(hits)
        else:
            if len(p_label) < 1:
                tp = 1
                fn = 0
            else:
                fn = 0
        return float(tp), float(fp), float(fn)
    
    def compute_overlap(self, recognized, ground_truth, bg_class=ignore_bg_class):
        p_label, p_start, p_end = get_labels_start_end_time(recognized, bg_class)
        y_label, y_start, y_end = get_labels_start_end_time(ground_truth, bg_class)

        hits_overlap = np.zeros(len(y_label))
        if len(y_label) > 0:
            for j in range(len(p_label)):
                intersection = np.minimum(p_end[j], y_end) - np.maximum(
                    p_start[j], y_start)
                union = np.maximum(p_end[j], y_end) - np.minimum(
                    p_start[j], y_start)
                IoU = (1.0 * intersection / union) * (
                    [p_label[j] == y_label[x] for x in range(len(y_label))])
                # Get the best scoring segment
                idx = np.array(IoU).argmax()

                if IoU[idx] >= hits_overlap[idx]:
                    hits_overlap[idx] = IoU[idx]
        return hits_overlap
    
    def _error_pass_f_score(self, before_hits_overlap, curr_hits_overlap, idx):
        t = self.overlap[idx]
        for b_p, c_p in zip(before_hits_overlap, curr_hits_overlap):
            if b_p >= t and c_p >= t:
                self.errpr_pass_count_dict[idx]['tt'] += 1
            elif b_p >= t and c_p < t:
                self.errpr_pass_count_dict[idx]['tf'] += 1
            elif b_p < t and c_p >= t:
                self.errpr_pass_count_dict[idx]['ft'] += 1
            elif b_p < t and c_p < t:
                self.errpr_pass_count_dict[idx]['ff'] += 1   
        
    def update(self, gt_labels, before_preds, curr_preds, bg_class=ignore_bg_class):
        for s in range(self.overlap_len):
            tp1, fp1, fn1 = self.f_score(before_preds, gt_labels, self.overlap[s], bg_class=bg_class)
            tp2, fp2, fn2 = self.f_score(curr_preds, gt_labels, self.overlap[s], bg_class=bg_class)

            # accumulate
            self.before_cls_tp[s] += tp1
            self.before_cls_fp[s] += fp1
            self.before_cls_fn[s] += fn1

            self.curr_cls_tp[s] += tp2
            self.curr_cls_fp[s] += fp2
            self.curr_cls_fn[s] += fn2
        
        before_hits_overlap = self.compute_overlap(before_preds, gt_labels, bg_class=bg_class)
        curr_hits_overlap = self.compute_overlap(curr_preds, gt_labels, bg_class=bg_class)
        
        for s in range(self.overlap_len - 1):
            self._error_pass_f_score(before_hits_overlap, curr_hits_overlap, s)

    def accumulate(self, idx):
        # calculate accuracy
        # before
        before_Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.before_cls_tp[s] / float(self.before_cls_tp[s] + self.before_cls_fp[s] + self.elps)
            recall = self.before_cls_tp[s] / float(self.before_cls_tp[s] + self.before_cls_fn[s] + self.elps)

            f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

            f1 = np.nan_to_num(f1) * 100
            before_Fscore[self.overlap[s]] = f1
        # curr
        curr_Fscore = dict()
        for s in range(self.overlap_len):
            precision = self.curr_cls_tp[s] / float(self.curr_cls_tp[s] + self.curr_cls_fp[s] + self.elps)
            recall = self.curr_cls_tp[s] / float(self.curr_cls_tp[s] + self.curr_cls_fn[s] + self.elps)

            f1 = 2.0 * (precision * recall) / (precision + recall + self.elps)

            f1 = np.nan_to_num(f1) * 100
            curr_Fscore[self.overlap[s]] = f1
        
        for i in range(self.overlap_len - 1):
            error_pass_rate = self.errpr_pass_count_dict[i]['ff'] / (self.errpr_pass_count_dict[i]['tf'] + self.errpr_pass_count_dict[i]['ff'] + self.elps)
            correction_capability = self.errpr_pass_count_dict[i]['ft'] / (self.errpr_pass_count_dict[i]['ft'] + self.errpr_pass_count_dict[i]['tt'] + self.elps)
            if idx == 0:
                self.table[i].add_row([0, before_Fscore[self.overlap[i]], before_Fscore[self.overlap[i + 1]], 0, 0])
                self.table[i].add_row([1, curr_Fscore[self.overlap[i]], curr_Fscore[self.overlap[i + 1]], error_pass_rate, correction_capability])
            else:
                self.table[i].add_row([idx + 1, curr_Fscore[self.overlap[i]], curr_Fscore[self.overlap[i + 1]], error_pass_rate, correction_capability])
    
    def print(self, log_file):
        for i in range(self.overlap_len - 1):
            print("Analysis Model Error Pass: \n" + str(self.table[i]) + "\n")
            with open(log_file, 'a') as f:
                f.write("Analysis Model Error Pass: \n" + str(self.table[i]) + "\n")

class ClsErrorPassAnalysis(object):
    def __init__(self,
                 num_classes) -> None:
        self.num_classes = num_classes
        self.before_matrix = np.zeros((num_classes, num_classes))
        self.curr_matrix = np.zeros((num_classes, num_classes))
        self.errpr_pass_count_dict = {'tt':0, 'tf':0, 'ff':0, 'ft':0}
        self.table = PrettyTable()
        self.table.field_names = ["model_idx", "Metric", "Error_pass_rate", "correction_capability"]

        self.elps = 1e-10
    
    def reset(self):
        self.before_matrix = np.zeros((self.num_classes, self.num_classes))
        self.curr_matrix = np.zeros((self.num_classes, self.num_classes))

    def update(self, gt_labels, before_preds, curr_preds):
        for b_p, t, c_p in zip(before_preds, gt_labels, curr_preds):
            self.before_matrix[b_p, t] += 1
            self.curr_matrix[c_p, t] += 1
            if b_p == t and t == c_p:
                self.errpr_pass_count_dict['tt'] += 1
            elif b_p == t and t != c_p:
                self.errpr_pass_count_dict['tf'] += 1
            elif b_p != t and t == c_p:
                self.errpr_pass_count_dict['ft'] += 1
            elif b_p != t and t != c_p:
                self.errpr_pass_count_dict['ff'] += 1   

    def accumulate(self, idx):
        # calculate accuracy
        # before
        sum_TP = 0
        n = np.sum(self.before_matrix)
        for i in range(self.num_classes):
            sum_TP += self.before_matrix[i, i]
        before_acc = sum_TP / n
        # curr
        sum_TP = 0
        n = np.sum(self.curr_matrix)
        for i in range(self.num_classes):
            sum_TP += self.curr_matrix[i, i]
        curr_acc = sum_TP / n
        
        error_pass_rate = self.errpr_pass_count_dict['ff'] / (self.errpr_pass_count_dict['tf'] + self.errpr_pass_count_dict['ff'] + self.elps)
        correction_capability = self.errpr_pass_count_dict['ft'] / (self.errpr_pass_count_dict['ft'] + self.errpr_pass_count_dict['tt'] + self.elps)
        if idx == 0:
            self.table.add_row([0, before_acc, 0, 0])
            self.table.add_row([1, curr_acc, error_pass_rate, correction_capability])
        else:
            self.table.add_row([idx + 1, curr_acc, error_pass_rate, correction_capability])
    
    def print(self, log_file):
        print("Analysis Model Error Pass: \n" + str(self.table) + "\n")
        with open(log_file, 'a') as f:
            f.write("Analysis Model Error Pass: \n" + str(self.table) + "\n")


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="convert pred and gt list to images.")
    parser.add_argument(
        "analysis_dirs_txt",
        type=str,
        help="path to a files you want to convert",
    )
    parser.add_argument(
        "action_dict_path",
        type=str,
        help="path to a action dict file",
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="path to output img",
        default="output"
    )

    return parser.parse_args()

def main() -> None:
    args = get_arguments()
    action_dict_path = args.action_dict_path

    # actions dict generate
    file_ptr = open(action_dict_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])
    
    analysis_dir_list = []
    with open(args.analysis_dirs_txt, 'r') as analysis_dir_files:
        for i, line in enumerate(analysis_dir_files):
            if i == 0:
                ground_truth_file = line.strip()
            else:
                analysis_dir_list.append(line.strip())
    
    cls_analysis = ClsErrorPassAnalysis(len(actions_dict))
    seg_analysis = SegErrorPassAnalysis()

    for before_analysis_dir, curr_anlysis_dir, idx in tqdm(zip(analysis_dir_list[:-1], analysis_dir_list[1:], range(len(analysis_dir_list) - 1)) ,desc='analysis model:'):
        analysis_filenames = os.listdir(before_analysis_dir)
        vid_list = ["-".join(vid.split('-')[:-1]) for vid in analysis_filenames if vid.endswith('pred.txt')]

        for vid in tqdm(vid_list, desc='analysis sample:'):
            # gt
            file_path = os.path.join(ground_truth_file, vid + '.txt')
            file_ptr = open(file_path, 'r')
            gt_labels = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            gt_labels_idx = np.array([actions_dict[name] for name in gt_labels])

            # before pred
            file_path = os.path.join(before_analysis_dir, vid + '-pred.txt')
            file_ptr = open(file_path, 'r')
            before_preds = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            before_preds_idx = np.array([actions_dict[name] for name in before_preds])

            # curr pred
            file_path = os.path.join(curr_anlysis_dir, vid + '-pred.txt')
            file_ptr = open(file_path, 'r')
            curr_preds = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            curr_preds_idx = np.array([actions_dict[name] for name in curr_preds])

            cls_analysis.update(gt_labels=gt_labels_idx, before_preds=before_preds_idx, curr_preds=curr_preds_idx)
            seg_analysis.update(gt_labels=gt_labels, before_preds=before_preds, curr_preds=curr_preds)
        cls_analysis.accumulate(idx)
        seg_analysis.accumulate(idx)
        cls_analysis.reset()
        seg_analysis.reset()
    
    # log
    cls_analysis.print(args.output_file)
    seg_analysis.print(args.output_file)


if __name__ == "__main__":
    main()