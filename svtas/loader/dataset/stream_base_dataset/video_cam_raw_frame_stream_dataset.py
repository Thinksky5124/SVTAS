'''
Author       : Thyssen Wen
Date         : 2022-10-23 11:11:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-08 10:01:19
Description  : Video CAM dataset class
FilePath     : /SVTAS/svtas/loader/dataset/stream_base_dataset/video_cam_raw_frame_stream_dataset.py
'''
import copy

import numpy as np
import torch

from svtas.utils import AbstractBuildFactory
from .raw_frame_stream_segmentation_dataset import \
    RawFrameStreamSegmentationDataset


@AbstractBuildFactory.register('dataset')
class RawFrameStreamCAMDataset(RawFrameStreamSegmentationDataset):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)
            
    def _get_one_videos_clip(self, idx, info):
        imgs_list = []
        labels_list = []
        masks_list = []
        vid_list = []
        raw_imgs_list = []
        precise_sliding_num_list = []
        
        for single_info in info:
            sample_segment = single_info.copy()
            sample_segment['sample_sliding_idx'] = idx
            sample_segment = self.pipeline(sample_segment)
            # imgs: tensor labels: ndarray mask: ndarray vid_list : str list
            imgs_list.append(copy.deepcopy(sample_segment['imgs'].unsqueeze(0)))
            raw_imgs_list.append(copy.deepcopy(sample_segment['raw_imgs']))
            labels_list.append(np.expand_dims(sample_segment['labels'], axis=0).copy())
            masks_list.append(np.expand_dims(sample_segment['masks'], axis=0).copy())
            vid_list.append(copy.deepcopy(sample_segment['video_name']))
            precise_sliding_num_list.append(np.expand_dims(sample_segment['precise_sliding_num'], axis=0).copy())

        imgs = copy.deepcopy(torch.concat(imgs_list, dim=0))
        raw_imgs = copy.deepcopy(raw_imgs_list)
        labels = copy.deepcopy(np.concatenate(labels_list, axis=0).astype(np.int64))
        masks = copy.deepcopy(np.concatenate(masks_list, axis=0).astype(np.float32))
        precise_sliding_num = copy.deepcopy(np.concatenate(precise_sliding_num_list, axis=0).astype(np.float32))

        # compose result
        data_dict = {}
        data_dict['imgs'] = imgs
        data_dict['labels'] = labels
        data_dict['masks'] = masks
        data_dict['raw_imgs'] = raw_imgs
        data_dict['precise_sliding_num'] = precise_sliding_num
        data_dict['vid_list'] = vid_list
        return data_dict
    
    def _get_end_videos_clip(self):
        # compose result
        data_dict = {}
        data_dict['imgs'] = 0
        data_dict['labels'] = 0
        data_dict['masks'] = 0
        data_dict['raw_imgs'] = 0
        data_dict['vid_list'] = []
        data_dict['sliding_num'] = 0
        data_dict['precise_sliding_num'] = 0
        data_dict['step'] = self.step_num
        data_dict['current_sliding_cnt'] = -1
        return data_dict