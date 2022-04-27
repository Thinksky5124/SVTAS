'''
Author: Thyssen Wen
Date: 2022-04-27 16:12:40
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 21:03:55
Description: file content
FilePath: /ETESVS/dataset/feature_pipline.py
'''
import numpy as np
import random
import torch
import copy
import torchvision.transforms as transforms
from .builder import PIPLINE

@PIPLINE.register()
class FeaturePipline():
    def __init__(self,
                 decode=None,
                 sample=None,
                 transform=None):
        self.decode = FeatureDecoder(**decode)
        self.sample = FeatureStreamSampler(**sample)
        self.transform = FeatureStreamTransform(transform)

    def __call__(self, results):
        # decode
        results = self.decode(results)
        # sample
        results = self.sample(results)
        # transform
        results = self.transform(results)
        return results

class FeatureDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend='numpy'):

        self.backend = backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'feature'

        feature = np.load(file_path)
        feature_len = feature.shape[-1]
        results['frames'] = feature
        results['frames_len'] = results['raw_labels'].shape[0]
        results['feature_len'] = feature_len
        
        return results

class FeatureFrameSample():
    def __init__(self, mode='random'):
        assert mode in ['random', 'uniform'], 'not support mode'
        self.mode = mode
    
    def random_sample(self, start_idx, end_idx, sample_rate):
        sample_idx = list(
                random.sample(list(range(start_idx, end_idx)),
                    len(list(range(start_idx, end_idx, sample_rate)))))
        sample_idx.sort()
        return sample_idx

    def uniform_sample(self, start_idx, end_idx, sample_rate):
        return list(range(start_idx, end_idx, sample_rate))
        
    def __call__(self, start_idx, end_idx, sample_rate):
        if self.mode == 'random':
            return self.random_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'uniform':
            return self.uniform_sample(start_idx, end_idx, sample_rate)
        else:
            raise NotImplementedError

class FeatureStreamSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 seg_len,
                 sample_rate=1,
                 clip_seg_num=15,
                 sliding_window=60,
                 ignore_index=-100,
                 sample_mode='random'
                 ):
        self.sample_rate = sample_rate
        self.seg_len = seg_len
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.sample = FeatureFrameSample(mode = sample_mode)

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        feature_len = int(results['feature_len'])
        results['frames_len'] = frames_len
        feature = results['frames']
        labels = results['raw_labels']

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len and end_frame < frames_len:
            vid_end_frame = end_frame
            if end_frame > feature_len:
                vid_end_frame = feature_len
            frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
            labels = labels[start_frame:end_frame].copy()
            frames_feature = feature[:, frames_idx]
            mask = np.ones((labels.shape[0]))
        elif start_frame < frames_len and end_frame >= frames_len:
            frames_idx = self.sample(start_frame, feature_len, self.sample_rate)
            labels = labels[start_frame:frames_len].copy()
            frames_feature = feature[:, frames_idx]
            pad_len = self.clip_seg_num - frames_feature.shape[-1]
            frames_feature = np.concatenate([frames_feature, np.zeros((2048, pad_len))], axis=-1)
            vaild_mask = np.ones((labels.shape[0]))
            mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
            void_mask = np.zeros((mask_pad_len))
            mask = np.concatenate([vaild_mask, void_mask], axis=0)
            labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])
        else:
            frames_feature = np.zeros((2048, self.clip_seg_num))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['feature'] = frames_feature[:, :self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        return results

class FeatureStreamTransform():
    def __init__(self, transform_list):
        transform_op_list = []
        for transforms_op in transform_list:
            name = list(transforms_op.keys())[0]
            if list(transforms_op.values())[0] is None:
                op = getattr(transforms, name)()
            else:
                op = getattr(transforms, name)(**list(transforms_op.values())[0])
            transform_op_list.append(op)
        self.imgs_transforms_pipeline = transforms.Compose(transform_op_list)

    def __call__(self, results):
        feature = results['feature'].astype(np.float32)
        feature = self.imgs_transforms_pipeline(feature).squeeze(0)
        results['feature'] = copy.deepcopy(feature)
        return results