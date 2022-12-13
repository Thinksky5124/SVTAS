'''
Author: Thyssen Wen
Date: 2022-03-21 11:12:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 16:44:35
Description: metric evaluate utils function
FilePath     : /SVTAS/svtas/metric/temporal_action_segmentation/temporal_action_segmentation_metric_utils.py
'''
import numpy as np

ignore_bg_class = ["background", "None"]

def get_labels_scores_start_end_time(input_np,
                                     frame_wise_labels,
                                     actions_dict,
                                     bg_class=ignore_bg_class):
    labels = []
    starts = []
    ends = []
    scores = []

    boundary_score_ptr = 0

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
                score = np.mean(
                        input_np[actions_dict[labels[boundary_score_ptr]], \
                            starts[boundary_score_ptr]:(ends[boundary_score_ptr] + 1)]
                        )
                scores.append(score)
                boundary_score_ptr = boundary_score_ptr + 1
            last_label = frame_wise_labels[i]
    if last_label not in bg_class:
        ends.append(i + 1)
        score = np.mean(
                    input_np[actions_dict[labels[boundary_score_ptr]], \
                        starts[boundary_score_ptr]:(ends[boundary_score_ptr] + 1)]
                    )
        scores.append(score)
        boundary_score_ptr = boundary_score_ptr + 1

    return labels, starts, ends, scores


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


def levenstein(p, y, norm=False):
    m_row = len(p)
    n_col = len(y)
    D = np.zeros([m_row + 1, n_col + 1])
    for i in range(m_row + 1):
        D[i, 0] = i
    for i in range(n_col + 1):
        D[0, i] = i

    for j in range(1, n_col + 1):
        for i in range(1, m_row + 1):
            if y[j - 1] == p[i - 1]:
                D[i, j] = D[i - 1, j - 1]
            else:
                D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1,
                              D[i - 1, j - 1] + 1)

    if norm:
        score = (1 - D[-1, -1] / (max(m_row, n_col) + 1e-6)) * 100
    else:
        score = D[-1, -1]

    return score


def edit_score(recognized, ground_truth, norm=True, bg_class=ignore_bg_class):
    P, _, _ = get_labels_start_end_time(recognized, bg_class)
    Y, _, _ = get_labels_start_end_time(ground_truth, bg_class)
    return levenstein(P, Y, norm)


def f_score(recognized, ground_truth, overlap, bg_class=ignore_bg_class):
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


