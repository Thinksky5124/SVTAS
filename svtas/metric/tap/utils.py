'''
Author       : Thyssen Wen
Date         : 2022-12-12 16:40:57
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 16:43:45
Description  : file content
FilePath     : /SVTAS/svtas/metric/temporal_action_proposal/utils.py
'''
import numpy as np
import pandas as pd

def boundary_AR(pred_boundary, gt_boundary, overlap_list, max_proposal):

    p_label, p_start, p_end, p_scores = pred_boundary
    y_label, y_start, y_end, _ = gt_boundary

    # sort proposal
    pred_dict = {
        "label": p_label,
        "start": p_start,
        "end": p_end,
        "scores": p_scores
    }
    pdf = pd.DataFrame(pred_dict)
    pdf = pdf.sort_values(by="scores", ascending=False)
    p_label = list(pdf["label"])
    p_start = list(pdf["start"])
    p_end = list(pdf["end"])
    p_scores = list(pdf["scores"])

    t_AR = np.zeros(len(overlap_list))

    # refine AN
    if len(p_label) < max_proposal and len(p_label) > 0:
        p_label = p_label + [p_label[-1]] * (max_proposal - len(p_label))
        p_start = p_start + [p_start[-1]] * (max_proposal - len(p_start))
        p_start = p_start + p_start[len(p_start) -
                                    (max_proposal - len(p_start)):]
        p_end = p_end + [p_end[-1]] * (max_proposal - len(p_end))
        p_scores = p_scores + [p_scores[-1]] * (max_proposal - len(p_scores))
    elif len(p_label) > max_proposal:
        p_label[max_proposal:] = []
        p_start[max_proposal:] = []
        p_end[max_proposal:] = []
        p_scores[max_proposal:] = []
    else:
        return np.mean(t_AR)

    for i in range(len(overlap_list)):
        overlap = overlap_list[i]

        tp = 0
        fp = 0

        if len(y_label) > 0:
            hits = np.zeros(len(y_label))

            for j in range(len(p_label)):
                intersection = np.minimum(p_end[j], y_end) - np.maximum(
                    p_start[j], y_start)
                union = np.maximum(p_end[j], y_end) - np.minimum(
                    p_start[j], y_start)
                IoU = (1.0 * intersection / union)
                # Get the best scoring segment
                idx = np.array(IoU).argmax()

                if IoU[idx] >= overlap and not hits[idx]:
                    tp += 1
                    hits[idx] = 1
                else:
                    fp += 1
            fn = len(y_label) - sum(hits)

            recall = float(tp) / (float(tp) + float(fn))
        else:
            if len(p_label) < 1:
                recall = float(1.0)
            else:
                recall = float(0.0)
        t_AR[i] = recall

    AR = np.mean(t_AR)
    return AR

