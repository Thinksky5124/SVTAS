'''
Author       : Thyssen Wen
Date         : 2022-11-01 12:25:27
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-08 09:51:38
Description  : video container
FilePath     : /SVTAS/svtas/loader/decode/container.py
'''
import re
import av
import cv2
import decord as de
import numpy as np
import copy

def get_container(backend):
    if backend == "numpy":
        return NPYContainer
    elif backend == "decord":
        return DecordContainer
    elif backend == "pyav":
        return PyAVContainer
    elif backend == "opencv":
        return OpenCVContainer
    else:
        raise NotImplementedError("Not support " + backend + "!")

class NPYContainer(object):
    def __init__(self, npy_file, temporal_dim=-1, is_transpose=False, revesive_name=[(r'(mp4|avi)', 'npy')]):
        self.temporal_dim = temporal_dim
        self.revesive_name = revesive_name
        for p, r in revesive_name:
            npy_file = re.sub(p, r, npy_file)
        if is_transpose:
            self.data = np.load(npy_file).T
        else:
            self.data = np.load(npy_file)
    
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

class DecordContainer(object):
    def __init__(self, video_file):
        self.data = de.VideoReader(video_file)

    def get_batch(self, frames_idx):
        return self.data.get_batch(frames_idx)
    
    def __len__(self):
        return len(self.data)

class PyAVContainer(object):
    """
    ref:https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/decoder.py
    """
    def __init__(self, video_file, multi_thread_decode=False):
        container = av.open(video_file)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        self.data = container
    
    def pyav_decode_stream(self, frames_index, stream, stream_name, buffer_size=0):
        """
        Decode the video with PyAV decoder.
        Args:
            stream (stream): PyAV stream.
            stream_name (dict): a dictionary of streams. For example, {"video": 0}
                means video stream at stream index 0.
            buffer_size (int): number of additional frames to decode beyond end_pts.
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

class OpenCVContainer(object):
    def __init__(self, video_file):
        self.data = cv2.VideoCapture(video_file)

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