'''
Author       : Thyssen Wen
Date         : 2022-11-01 12:25:27
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 14:11:35
Description  : video container
FilePath     : /SVTAS/svtas/loader/decode/container.py
'''
import re
import av
import cv2
import decord as de
import numpy as np
import copy
from mvextractor.videocap import VideoCap as MVVideoCap
from ..builder import CONTAINER

@CONTAINER.register()
class NPYContainer(object):
    def __init__(self, file_path, temporal_dim=-1, is_transpose=False, revesive_name=[(r'(mp4|avi)', 'npy')]):
        self.temporal_dim = temporal_dim
        self.revesive_name = revesive_name
        for p, r in revesive_name:
            file_path = re.sub(p, r, file_path)
        if is_transpose:
            self.data = np.load(file_path).T
        else:
            self.data = np.load(file_path)
        self.out_dtype = 'numpy'
    
    def concat(self, rhs_container, dim=0):
        self.data = np.concatenate([self.data, rhs_container.data], axis=dim)
        return self

    def get_batch(self, frames_idx):
        if self.temporal_dim == -1:
            return self.data[:, frames_idx]
        else:
            return self.data[frames_idx, :]
    
    def __len__(self):
        return self.data.shape[self.temporal_dim]

@CONTAINER.register()
class DecordContainer(object):
    def __init__(self, file_path):
        self.data = de.VideoReader(file_path)
        self.out_dtype = 'PIL'

    def get_batch(self, frames_idx):
        return self.data.get_batch(frames_idx)
    
    def __len__(self):
        return len(self.data)

@CONTAINER.register()
class PyAVContainer(object):
    """
    ref:https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/decoder.py
    """
    def __init__(self, file_path, multi_thread_decode=False):
        container = av.open(file_path)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        self.data = container
        self.out_dtype = 'numpy'
    
    def pyav_decode_stream(self, frames_index, stream, stream_name):
        """
        Decode the video with PyAV decoder.
        Args:
            stream (stream): PyAV stream.
            stream_name (dict): a dictionary of streams. For example, {"video": 0}
                means video stream at stream index 0.
        Returns:
            result (list): list of frames decoded.
            max_pts (int): max Presentation TimeStamp of the video sequence.
        """
        # Seeking in the stream is imprecise. Thus, seek to an ealier PTS by a
        # margin pts.
        margin = 1024
        seek_offset = max(min(frames_index) - margin, 0)

        self.data.seek(seek_offset, any_frame=False, backward=True, stream=stream)
        frames = {}
        max_pts = 0
        for frame in self.data.decode(**stream_name):
            max_pts = max(max_pts, frame.pts)
            # frame.pts is start from 1
            if (frame.pts - 1) in frames_index:
                frames[frame.pts] = frame
            if len(frames) == len(frames_index):
                break
        result = [frames[pts] for pts in sorted(frames)]
        return result, max_pts
    
    def pyav_decode(self, frames_index):
        """
        Convert the video from its original fps to the target_fps. If the video
        support selective decoding (contain decoding information in the video head),
        the perform temporal selective decoding and sample a clip from the video
        with the PyAV decoder. If the video does not support selective decoding,
        decode the entire video.
        Args:
            frames_index (list[int]): frame index to be sampled
        Returns:
            frames (tensor): decoded frames from the video. Return None if the no
                video stream was found.
        """

        frames = None
        # If video stream was found, fetch video frames from the video.
        if self.data.streams.video:
            video_frames, max_pts = self.pyav_decode_stream(
                frames_index,
                self.data.streams.video[0],
                {"video": 0},
            )
            self.data.close()

            frames = [frame.to_rgb().to_ndarray() for frame in video_frames]
            frames = np.stack(frames)
        return frames

    def get_batch(self, frames_idx):
        return self.pyav_decode(frames_idx)
    
    def __len__(self):
        return self.data.streams.video[0].frames

@CONTAINER.register()
class OpenCVContainer(object):
    def __init__(self, file_path):
        self.data = cv2.VideoCapture(file_path)
        self.out_dtype = 'numpy'

    def get_batch(self, frames_idx):
        frames = []
        margin = 128
        current_frame_idx = max(0, min(frames_idx) - margin)
        start_frame_idx = current_frame_idx
        self.data.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        for i in range(start_frame_idx, len(self)):
            ret, img = self.data.read()
            if ret:
                if current_frame_idx in frames_idx:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_img)
            else:
                break
            current_frame_idx += 1
            if len(frames) == len(frames_idx):
                break
        frames = copy.deepcopy(np.stack(frames))
        self.data.release()
        return frames
    
    def __len__(self):
        return int(self.data.get(cv2.CAP_PROP_FRAME_COUNT))

@CONTAINER.register()
class MVExtractor(object):
    def __init__(self, file_path, need_residual=True, need_mvs=True, argument=False):
        video = cv2.VideoCapture(file_path)
        self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.data = MVVideoCap.open(file_path)
        self.argument = argument
        self.need_residual = need_residual
        self.need_mvs = need_mvs
        self.out_dtype = 'dict'

    def get_batch(self, frames_idx):
        frames = []
        mv_flows = []
        margin = 128
        current_frame_idx = max(0, min(frames_idx) - margin)
        start_frame_idx = current_frame_idx
        self.data.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        for i in range(start_frame_idx, len(self)):
            ret, img = self.data.read()
            ret, img, motion_vectors, frame_type, timestamp = self.data.read() 
            if ret:
                if current_frame_idx in frames_idx:
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    frames.append(rgb_img)
                    
                    mv_frame = np.zeros((rgb_img.shape[0], rgb_img.shape[1], 2))
                    if len(motion_vectors) > 0:
                        num_mvs = np.shape(motion_vectors)[0]
                        for mv in np.split(motion_vectors, num_mvs):
                            block_w = mv[0, 1]
                            block_h = mv[0, 2]
                            block_x = mv[0, 3] // block_w
                            block_y = mv[0, 4] // block_h
                            mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 0] = mv[0, 3] - mv[0, 5]
                            mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 1] = mv[0, 4] - mv[0, 6]
                    mv_flows.append(mv_frame)

                    pad_factor = 32
                    res_img = np.full_like(frame, 128)
                    if last_frame is None:
                        last_frame = frame
                        last_frame = cv2.copyMakeBorder(last_frame, pad_factor, pad_factor, pad_factor, pad_factor, cv2.BORDER_CONSTANT, value=(128,128,128))
                    else:
                        mv_compress = copy.deepcopy(last_frame)
                        w = frame.shape[1] + pad_factor
                        h = frame.shape[0] + pad_factor
                        if len(motion_vectors) > 0:
                            num_mvs = np.shape(motion_vectors)[0]
                            for mv in np.split(motion_vectors, num_mvs):
                                block_w = mv[0, 1]
                                block_h = mv[0, 2]
                                block_x = mv[0, 3] // block_w
                                block_y = mv[0, 4] // block_h

                                dst_x_min = pad_factor + block_x * block_w + mv[0, 5] - mv[0, 3]
                                dst_x_max = pad_factor + (block_x + 1) * block_w + mv[0, 5] - mv[0, 3]
                                dst_y_min = pad_factor + block_y * block_h + mv[0, 6] - mv[0, 4]
                                dst_y_max = pad_factor + (block_y + 1) * block_h + mv[0, 6] - mv[0, 4]

                                src_x_min = pad_factor + block_x * block_w
                                src_x_max = pad_factor + (block_x + 1) * block_w
                                src_y_min = pad_factor + block_y * block_h
                                src_y_max = pad_factor + (block_y + 1) * block_h

                                mv_compress[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = last_frame[src_y_min:src_y_max, src_x_min:src_x_max]
                        res_img = frame - mv_compress[pad_factor:h, pad_factor:w]

                        res_img = (res_img + (res_img == 0) * 128).astype(np.uint8)
                        # res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2GRAY)
                        # res_img=cv2.medianBlur(res_img, 3)
                        # res_img=cv2.GaussianBlur(res_img,(3,3),0,1)

                        last_frame = cv2.copyMakeBorder(frame, pad_factor, pad_factor, pad_factor, pad_factor, cv2.BORDER_CONSTANT, value=(128,128,128))
                        
                    return res_img, last_frame
            else:
                break
            current_frame_idx += 1
            if len(frames) == len(frames_idx):
                break
        frames = copy.deepcopy(np.stack(frames))
        flows = copy.deepcopy(np.stack(mv_flows))
        self.data.release()
        return frames, flows, residual
    
    def __len__(self):
        return self.len