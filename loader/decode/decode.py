'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:26:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:29:15
Description  : feature decode
FilePath     : /ETESVS/loader/decode/decode.py
'''
import numpy as np
import decord as de
from ..builder import DECODE

@DECODE.register()
class FeatureDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend='numpy',
                 is_transpose=False):

        self.backend = backend
        self.is_transpose = is_transpose

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'feature'

        feature = np.load(file_path)
        if self.is_transpose is True:
            feature = feature.T
        feature_len = feature.shape[-1]
        results['frames'] = feature
        results['frames_len'] = results['raw_labels'].shape[0]
        results['feature_len'] = feature_len
        
        return results

@DECODE.register()
class VideoDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend='decord'):

        self.backend = backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'video'
        results['backend'] = self.backend

        container = de.VideoReader(file_path)
        video_len = len(container)
        results['frames'] = container
        results['frames_len'] = results['raw_labels'].shape[0]
        results['video_len'] = video_len
        
        return results

@DECODE.register()
class RGBFlowVideoDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend='decord'):

        self.backend = backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        flow_path = results['flow_path']
        results['format'] = 'video'
        results['backend'] = self.backend

        rgb_container = de.VideoReader(file_path)
        flow_container = de.VideoReader(flow_path)
        video_len = len(rgb_container)
        results['rgb_frames'] = rgb_container
        results['flow_frames'] = flow_container
        results['frames_len'] = results['raw_labels'].shape[0]
        results['video_len'] = video_len
        
        return results