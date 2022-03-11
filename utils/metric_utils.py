import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
        score = (1 - D[-1, -1] / max(m_row, n_col)) * 100
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


# ref: https://github.com/activitynet/ActivityNet/blob/master/Evaluation/utils.py
def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap


# ref: https://github.com/activitynet/ActivityNet/blob/master/Evaluation/utils.py
def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU


# ref: https://github.com/activitynet/ActivityNet/blob/master/Evaluation/eval_detection.py
def compute_average_precision_detection(ground_truth,
                                        prediction,
                                        tiou_thresholds=np.linspace(
                                            0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't_start', 't_end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't_start', 't_end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(
                this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t_start', 't_end']].values,
                               this_gt[['t_start', 't_end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :],
                                         recall_cumsum[tidx, :])

    return ap


def get_predictions_with_label(prediction_by_label, label_name, cidx):
    """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
    try:
        return prediction_by_label.get_group(label_name).reset_index(drop=True)
    except:
        return pd.DataFrame()


def wrapper_compute_average_precision(prediction, ground_truth, tiou_thresholds,
                                      activity_index):
    """Computes average precision for each class in the subset.
        """
    activity_dict = activity_index.copy()
    # del background class
    for label_name in list(activity_dict.keys()):
        if label_name in ignore_bg_class:
            del activity_dict[label_name]

    ap = np.zeros((len(tiou_thresholds), len(activity_dict)))

    # Adaptation to query faster
    ground_truth_by_label = ground_truth.groupby('label')
    prediction_by_label = prediction.groupby('label')

    results = Parallel(n_jobs=len(activity_dict))(
        delayed(compute_average_precision_detection)(
            ground_truth=get_predictions_with_label(ground_truth_by_label,
                                                    label_name, cidx),
            prediction=get_predictions_with_label(prediction_by_label,
                                                  label_name, cidx),
            tiou_thresholds=tiou_thresholds,
        ) for label_name, cidx in activity_dict.items())

    for i, cidx in enumerate(activity_dict.values()):
        ap[:, cidx] = results[i]

    return ap
