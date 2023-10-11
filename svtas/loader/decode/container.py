'''
Author       : Thyssen Wen
Date         : 2022-11-01 12:25:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 20:58:28
Description  : video container
FilePath     : /SVTAS/svtas/loader/decode/container.py
'''
import re
import cv2
import decord as de
import numpy as np
import copy
# from mvextractor.videocap import VideoCap as MVVideoCap
from svtas.utils import AbstractBuildFactory
from svtas.utils import is_av_available

if is_av_available():
    import av

@AbstractBuildFactory.register('sample_container')
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

@AbstractBuildFactory.register('sample_container')
class DecordContainer(object):
    def __init__(self, file_path, to_ndarray=False, sample_dim=2):
        self.data = de.VideoReader(file_path)
        self.out_dtype = 'PIL'
        self.to_ndarray = to_ndarray
        if to_ndarray is True:
            self.out_dtype = 'numpy'
            self.sample_dim = sample_dim

    def get_batch(self, frames_idx):
        if self.to_ndarray:
            return self.data.get_batch(frames_idx).asnumpy()[:, :, :, 1:(self.sample_dim+1)]
        else:
            return self.data.get_batch(frames_idx)

    def __len__(self):
        return len(self.data)

@AbstractBuildFactory.register('sample_container')
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

@AbstractBuildFactory.register('sample_container')
class OpenCVContainer(object):
    def __init__(self, file_path):
        self.data = cv2.VideoCapture(file_path)
        self.out_dtype = 'numpy'

    def get_batch(self, frames_idx):
        frames = []
        margin = 1024
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

# @AbstractBuildFactory.register('sample_container')
# class MVExtractor(object):
#     def __init__(self,
#                  file_path,
#                  need_residual=True,
#                  need_mvs=True,
#                  argument=False):
#         video = cv2.VideoCapture(file_path)
#         self.len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
#         self.data = MVVideoCap()
#         ret = self.data.open(file_path)
#         if not ret:
#             raise RuntimeError(f"Could not open {file_path}")

#         self.argument = argument
#         self.need_residual = need_residual
#         self.need_mvs = need_mvs
#         self.out_dtype = 'dict'
#         self.dict_keys = ['imgs']
#         if need_mvs:
#             self.dict_keys.append('flows')
#         if need_residual:
#             self.dict_keys.append('res')

#         self.last_frame = None
#         self.last_mvs_frame = None
#         self.pad_factor = 32
    
#     def _get_mvs_img(self, mvs, w, h, output_dict):
#         mv_frame = np.zeros((h, w, 2))
#         if len(mvs) > 0:
#             num_mvs = np.shape(mvs)[0]
#             for mv in np.split(mvs, num_mvs):
#                 block_w = mv[0, 1]
#                 block_h = mv[0, 2]
#                 block_x = mv[0, 3] // block_w
#                 block_y = mv[0, 4] // block_h
#                 mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 0] = mv[0, 5] - mv[0, 3]
#                 mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 1] = mv[0, 6] - mv[0, 4]
#         if 'flows' not in output_dict.keys():
#             output_dict['flows'] = [mv_frame]
#         else:
#             output_dict['flows'].append(mv_frame)
#         return output_dict

#     def _get_res_img(self, img, mvs, output_dict):
#         res_img = np.full_like(img, 0)
#         if self.last_frame is None:
#             self.last_frame = img
#             self.last_frame = cv2.copyMakeBorder(self.last_frame, self.pad_factor, self.pad_factor, self.pad_factor, self.pad_factor, cv2.BORDER_CONSTANT, value=(0,0,0))
#         else:
#             mv_compress = copy.deepcopy(self.last_frame)
#             w = img.shape[1] + self.pad_factor
#             h = img.shape[0] + self.pad_factor
#             if len(mvs) > 0:
#                 num_mvs = np.shape(mvs)[0]
#                 for mv in np.split(mvs, num_mvs):
#                     block_w = mv[0, 1]
#                     block_h = mv[0, 2]
#                     block_x = max(min(mv[0][3] // block_w, img.shape[1] // block_w), 0)
#                     block_y = max(min(mv[0][4] // block_h, img.shape[0] // block_h), 0)

#                     dst_x_min = self.pad_factor + block_x * block_w + mv[0, 5] - mv[0, 3]
#                     dst_x_max = self.pad_factor + (block_x + 1) * block_w + mv[0, 5] - mv[0, 3]
#                     dst_y_min = self.pad_factor + block_y * block_h + mv[0, 6] - mv[0, 4]
#                     dst_y_max = self.pad_factor + (block_y + 1) * block_h + mv[0, 6] - mv[0, 4]

#                     src_x_min = self.pad_factor + block_x * block_w
#                     src_x_max = self.pad_factor + (block_x + 1) * block_w
#                     src_y_min = self.pad_factor + block_y * block_h
#                     src_y_max = self.pad_factor + (block_y + 1) * block_h

#                     mv_compress[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = self.last_frame[src_y_min:src_y_max, src_x_min:src_x_max]
#             res_img = img - mv_compress[self.pad_factor:h, self.pad_factor:w]
#             res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
#             self.last_frame = cv2.copyMakeBorder(img, self.pad_factor, self.pad_factor, self.pad_factor, self.pad_factor, cv2.BORDER_CONSTANT, value=(0,0,0))
#         if 'res' not in output_dict.keys():
#             output_dict['res'] = [res_img]
#         else:
#             output_dict['res'].append(res_img)
#         return output_dict

#     def _get_rgb_img(self, img, output_dict):
#         rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         output_dict['imgs'].append(rgb_img)
#         return output_dict

#     def get_batch(self, frames_idx):
#         output_dict = {'imgs':[]}
#         current_frame_idx = 0
#         for i in range(0, len(self)):
#             ret, img, motion_vectors, frame_type, timestamp = self.data.read() 
#             if ret:
#                 if current_frame_idx in frames_idx:
#                     h, w = img.shape[0], img.shape[1]
#                     output_dict = self._get_rgb_img(img=img, output_dict=output_dict)
#                     if self.need_mvs:
#                         output_dict = self._get_mvs_img(mvs=motion_vectors, w=w, h=h, output_dict=output_dict)
#                     if self.need_residual:
#                         output_dict = self._get_res_img(img=img, mvs=motion_vectors, output_dict=output_dict)
#             else:
#                 break
#             current_frame_idx += 1
#             if len(output_dict['imgs']) == len(frames_idx):
#                 break
#         return_dict = {}
#         for k, v in output_dict.items():
#             frames = copy.deepcopy(np.stack(v))
#             return_dict[k] = frames
#         self.data.release()
#         return return_dict
    
#     def __len__(self):
#         return self.len

@AbstractBuildFactory.register('sample_container')
class PyAVMVExtractor(object):
    def __init__(self,
                 file_path,
                 need_residual=True,
                 need_mvs=True,
                 argument=False,
                 multi_thread_decode=False,
                 bound=100):
        container = av.open(file_path)
        if multi_thread_decode:
            # Enable multiple threads for decoding.
            container.streams.video[0].thread_type = "AUTO"
        container.streams.video[0].export_mvs = True
        self.data = container
        self.argument = argument
        self.need_residual = need_residual
        self.need_mvs = need_mvs
        self.bound = bound
        self.out_dtype = 'dict'
        self.dict_keys = ['imgs']
        if need_mvs:
            self.dict_keys.append('flows')
        if need_residual:
            self.dict_keys.append('res')

        self.last_frame = None
        self.last_mvs_frame = None
        self.pad_factor = 32
    
    def _get_mvs_img(self, mvs, w, h, output_dict):
        mv_frame = np.zeros((h, w, 2))
        if len(mvs) > 0:
            num_mvs = np.shape(mvs)[0]
            for mv in np.split(mvs, num_mvs):
                block_w = mv[0][1]
                block_h = mv[0][2]
                block_x = mv[0][3] // block_w
                block_y = mv[0][4] // block_h
                mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 0] = (mv[0][8] / mv[0][10]) * mv[0][0]
                mv_frame[(block_y * block_h):((block_y + 1) * block_h), (block_x * block_w):((block_x + 1) * block_w), 1] = (mv[0][9] / mv[0][10]) * mv[0][0]
        if 'flows' not in output_dict.keys():
            output_dict['flows'] = [mv_frame]
        else:
            output_dict['flows'].append(mv_frame)
        return output_dict

    def _get_res_img(self, img, mvs, output_dict):
        res_img = np.full_like(img, 0)
        if self.last_frame is None:
            self.last_frame = img
            self.last_frame = cv2.copyMakeBorder(self.last_frame, self.pad_factor, self.pad_factor, self.pad_factor, self.pad_factor, cv2.BORDER_CONSTANT, value=(0,0,0))
        else:
            mv_compress = copy.deepcopy(self.last_frame)
            w = img.shape[1] + self.pad_factor
            h = img.shape[0] + self.pad_factor
            if len(mvs) > 0:
                num_mvs = np.shape(mvs)[0]
                for mv in np.split(mvs, num_mvs):
                    block_w = mv[0][1]
                    block_h = mv[0][2]
                    block_x = max(min(mv[0][3] // block_w, img.shape[1] // block_w), 0)
                    block_y = max(min(mv[0][4] // block_h, img.shape[0] // block_h), 0)

                    dst_x_min = self.pad_factor + block_x * block_w + mv[0][5] - mv[0][3]
                    dst_x_max = self.pad_factor + (block_x + 1) * block_w + mv[0][5] - mv[0][3]
                    dst_y_min = self.pad_factor + block_y * block_h + mv[0][6] - mv[0][4]
                    dst_y_max = self.pad_factor + (block_y + 1) * block_h + mv[0][6] - mv[0][4]

                    src_x_min = self.pad_factor + block_x * block_w
                    src_x_max = self.pad_factor + (block_x + 1) * block_w
                    src_y_min = self.pad_factor + block_y * block_h
                    src_y_max = self.pad_factor + (block_y + 1) * block_h

                    mv_compress[dst_y_min:dst_y_max, dst_x_min:dst_x_max] = self.last_frame[src_y_min:src_y_max, src_x_min:src_x_max]

            res_img = img.astype(np.float16) - mv_compress[self.pad_factor:h, self.pad_factor:w].astype(np.float16)
            res_img = (res_img + self.bound) * (255.0 / (2 * self.bound))
            res_img = res_img.round().astype(np.uint8)

            res_img = cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB)
            self.last_frame = cv2.copyMakeBorder(img, self.pad_factor, self.pad_factor, self.pad_factor, self.pad_factor, cv2.BORDER_CONSTANT, value=(0,0,0))
        if 'res' not in output_dict.keys():
            output_dict['res'] = [res_img]
        else:
            output_dict['res'].append(res_img)
        return output_dict

    def _get_rgb_img(self, img, output_dict):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output_dict['imgs'].append(rgb_img)
        return output_dict

    def get_batch(self, frames_idx):
        output_dict = {'imgs':[]}
        margin = 1024
        seek_offset = max(min(frames_idx) - margin, 0)

        self.data.seek(seek_offset, any_frame=False, backward=True)
        for pi, packet in enumerate(self.data.demux()):
            if packet.stream.type == 'video':
                for fi, frame in enumerate(packet.decode()):
                    if frame.index in frames_idx:
                        img = frame.to_ndarray(format='bgr24')
                        h, w = img.shape[0], img.shape[1]
                        output_dict = self._get_rgb_img(img=img, output_dict=output_dict)
                        for di, data in enumerate(frame.side_data):
                            if data.type.value == 8:
                                motion_vectors = data.to_ndarray()
                                if self.need_mvs:
                                    output_dict = self._get_mvs_img(mvs=motion_vectors, w=w, h=h, output_dict=output_dict)
                                if self.need_residual:
                                    output_dict = self._get_res_img(img=img, mvs=motion_vectors, output_dict=output_dict)
                    if len(output_dict['imgs']) == len(frames_idx):
                        break
            if len(output_dict['imgs']) == len(frames_idx):
                break
        return_dict = {}
        for k, v in output_dict.items():
            frames = copy.deepcopy(np.stack(v))
            return_dict[k] = frames
        self.data.close()
        return return_dict
    
    def __len__(self):
        return self.data.streams.video[0].frames