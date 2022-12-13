'''
Author       : Thyssen Wen
Date         : 2022-12-12 16:41:02
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-12 16:44:10
Description  : file content
FilePath     : /SVTAS/svtas/metric/temporal_action_localization/utils.py
'''
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ignore_bg_class = ["background", "None"]

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
    if prediction.empty or ground_truth.empty:
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

    results = Parallel(n_jobs=4)(
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