'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:26:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 12:57:40
Description  : feature decode
FilePath     : /SVTAS/svtas/loader/decode/decode.py
'''
import numpy as np

from ..builder import DECODE

from .container import get_container

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
        if "flow_feature_name" in list(results.keys()):
            flow_feature = np.load(results['flow_feature_name'])
            feature = np.concatenate([feature, flow_feature], axis=0)
        feature_len = feature.shape[1]
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
        
        try:
            container = get_container(self.backend)(file_path)
        except:
            print("file: " + file_path + " get error!")
            raise
        video_len = len(container)
        results['frames'] = container
        results['frames_len'] = results['raw_labels'].shape[0]
        results['video_len'] = video_len
        
        return results

    
@DECODE.register()
class FlowVideoDecoder(object):
    """
    get flow from file
    """
    def __init__(self,
                 backend='numpy'):
        self.backend = backend

    def __call__(self, results):
        file_path = results['filename']
        results['format'] = 'video'
        results['backend'] = self.backend

        try:
            container = get_container(self.backend)(file_path)
        except:
            print("file: " + file_path + " get error!")
            raise
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

        try:
            rgb_container = get_container(self.backend)(file_path)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        try:
            flow_container = get_container(self.backend)(flow_path)
        except:
            print("file: " + flow_path + " get error!")
            raise
        video_len = len(rgb_container)
        results['rgb_frames'] = rgb_container
        results['flow_frames'] = flow_container
        results['frames_len'] = results['raw_labels'].shape[0]
        results['video_len'] = video_len
        
        return results