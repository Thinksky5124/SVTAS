'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:26:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 14:22:37
Description  : feature decode
FilePath     : /SVTAS/svtas/loader/decode/decode.py
'''
from ..builder import DECODE, build_container

@DECODE.register()
class FeatureDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=0,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 ),
                 flow_feature_backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=0,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )):

        self.backend = backend
        self.flow_feature_backend = flow_feature_backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'feature'
        self.backend['file_path'] = file_path
        try:
            feature_container = build_container(self.backend)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        if "flow_feature_name" in list(results.keys()) and self.flow_feature_backend is not None:
            self.flow_feature_backend['file_path'] = results['flow_feature_name']
            try:
                flow_container = build_container(self.flow_feature_backend)
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
                 backend=dict(
                    name='DecordContainer')):

        self.backend = backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        results['format'] = 'video'
        self.backend['file_path'] = file_path
        
        try:
            container = build_container(self.backend)
        except:
            print("file: " + file_path + " get error!")
            raise
        video_len = len(container)
        results['frames'] = container
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['video_len'] = video_len
        
        return results

@DECODE.register()
class TwoPathwayVideoDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 rgb_backend=dict(
                    name='DecordContainer'
                 ),
                 flow_backend=dict(
                    name='NPYContainer',
                    temporal_dim=0,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )):

        self.rgb_backend = rgb_backend
        self.flow_backend =flow_backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        flow_path = results['flows_path']
        results['format'] = 'video'
        self.rgb_backend['file_path'] = file_path
        self.flow_backend['file_path'] = flow_path

        try:
            rgb_container = build_container(self.rgb_backend)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        try:
            flow_container = build_container(self.flow_backend)
        except:
            print("file: " + flow_path + " get error!")
            raise
        video_len = len(rgb_container)
        results['rgb_frames'] = rgb_container
        results['flow_frames'] = flow_container
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['video_len'] = video_len
        
        return results

@DECODE.register()
class ThreePathwayVideoDecoder():
    """
    Decode mp4 file to frames.
    Args:
        filepath: the file path of mp4 file
    """
    def __init__(self,
                 rgb_backend=dict(
                    name='DecordContainer'
                 ),
                 flow_backend=dict(
                    backend=dict(
                    name='DecordContainer',
                    to_ndarray=True,
                    sample_dim=2)
                 ),
                 res_backend=dict(
                    backend=dict(
                    name='DecordContainer')
                 )):

        self.rgb_backend = rgb_backend
        self.flow_backend =flow_backend
        self.res_backend = res_backend

    def __call__(self, results):
        """
        Perform mp4 decode operations.
        return:
            List where each item is a numpy array after decoder.
        """
        file_path = results['filename']
        flow_path = results['flows_path']
        res_path = results['res_path']
        results['format'] = 'video'
        self.rgb_backend['file_path'] = file_path
        self.flow_backend['file_path'] = flow_path
        self.res_backend['file_path'] = res_path

        try:
            rgb_container = build_container(self.rgb_backend)
        except:
            print("file: " + file_path + " get error!")
            raise
        
        try:
            flow_container = build_container(self.flow_backend)
        except:
            print("file: " + flow_path + " get error!")
            raise

        try:
            res_container = build_container(self.flow_backend)
        except:
            print("file: " + flow_path + " get error!")
            raise

        video_len = len(rgb_container)
        results['rgb_frames'] = rgb_container
        results['flow_frames'] = flow_container
        results['res_frames'] = res_container
        results['frames_len'] = int(results['raw_labels'].shape[0])
        results['video_len'] = video_len
        
        return results