'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:26:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-07 23:27:28
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
                 is_transpose=False,
                 temporal_dim=0,
                 revesive_name=[(r'(mp4|avi)', 'npy')]):

        self.backend = backend
        self.is_transpose = is_transpose
        self.temporal_dim = temporal_dim
        self.revesive_name = revesive_name

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'feature'

        try:
            feature_container = get_container(self.backend)(file_path, temporal_dim=self.temporal_dim, is_transpose=self.is_transpose, revesive_name=self.revesive_name)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        if "flow_feature_name" in list(results.keys()):
            try:
                flow_container = get_container(self.backend)(results['flow_feature_name'])
                feature_container = feature_container.concat(flow_container, dim=0)
            except:
                print("file: " + file_path + " get error!")
                raise

        feature_len = len(feature_container)
        results['frames'] = feature_container
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['feature_len'] = int(feature_len)
        
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
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['video_len'] = video_len
        
        return results

    
@DECODE.register()
class FlowVideoDecoder(object):
    """
    get flow from file
    """
    def __init__(self,
                 backend='numpy',
                 temporal_dim=0,
                 revesive_name=[(r'(mp4|avi)', 'npy')]):
        self.backend = backend
        self.temporal_dim = temporal_dim
        self.revesive_name = revesive_name

    def __call__(self, results):
        file_path = results['filename']
        results['format'] = 'video'
        results['backend'] = self.backend

        try:
            container = get_container(self.backend)(file_path, temporal_dim=self.temporal_dim, revesive_name=self.revesive_name)
        except:
            print("file: " + file_path + " get error!")
            raise
        video_len = len(container)
        results['frames'] = container
        results['frames_len'] = int(results['raw_labels'].shape[0])
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
                 rgb_backend='decord',
                 flow_backend='numpy',
                 flow_temporal_dim=0,
                 flow_revesive_name=[(r'(mp4|avi)', 'npy')]):

        self.rgb_backend = rgb_backend
        self.flow_backend =flow_backend
        self.temporal_dim = flow_temporal_dim
        self.revesive_name = flow_revesive_name

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        flow_path = results['flows_path']
        results['format'] = 'video'
        results['rgb_backend'] = self.rgb_backend
        results['flow_backend'] = self.flow_backend

        try:
            rgb_container = get_container(self.rgb_backend)(file_path)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        try:
            flow_container = get_container(self.flow_backend)(flow_path, temporal_dim=self.temporal_dim, revesive_name=self.revesive_name)
        except:
            print("file: " + flow_path + " get error!")
            raise
        video_len = len(rgb_container)
        results['rgb_frames'] = rgb_container
        results['flow_frames'] = flow_container
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['video_len'] = video_len
        
        return results