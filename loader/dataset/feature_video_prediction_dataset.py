'''
Author       : Thyssen Wen
Date         : 2022-05-18 16:14:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 16:16:46
Description  : Feature Video Prediction dataset
FilePath     : /ETESVS/loader/dataset/feature_video_prediction_dataset.py
'''
import numpy as np
import copy
import torch
from .feature_segmentation_dataset import FeatureSegmentationDataset
from ..builder import DATASET


@DATASET.register()
class FeatureVideoPredictionDataset(FeatureSegmentationDataset):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
    
    def _get_one_videos_clip(self, idx, info):
        feature_list = []
        labels_list = []
        pred_labels_list = []
        masks_list = []
        vid_list = []
        precise_sliding_num_list = []

        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment['sample_sliding_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            feature_list.append(copy.deepcopy(sample_segment['feature'].unsqueeze(0)))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            pred_labels_list.append(np.expand_dims(sample_segment['pred_labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['mask'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))
            precise_sliding_num_list.append(np.expand_dims(sample_segment['precise_sliding_num'], axis=0).copy())

        feature = copy.deepcopy(torch.concat(feature_list, dim=0))
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        pred_labels = copy.deepcopy(np.concatenate(pred_labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        precise_sliding_num = copy.deepcopy(np.concatenate(precise_sliding_num_list, axis=0).astype(np.float32))

        # compose result
        data_dict = {}
        data_dict['feature'] = feature
        data_dict['labels'] = labels
        data_dict['pred_labels'] = pred_labels
        data_dict['masks'] = masks
        data_dict['precise_sliding_num'] = precise_sliding_num
        data_dict['vid_list'] = vid_list
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['feature'] = 0
        data_dict['labels'] = 0
        data_dict['pred_labels'] = 0
        data_dict['masks'] = 0
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['precise_sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict