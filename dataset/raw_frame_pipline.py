'''
Author: Thyssen Wen
Date: 2022-03-18 19:25:14
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-11 12:07:36
Description: data prepare pipline function
FilePath     : /ETESVS/dataset/raw_frame_pipline.py
'''
import torchvision.transforms as transforms
import decord as de
import numpy as np
import random
import torch
import copy
from PIL import Image
from .builder import PIPLINE

@PIPLINE.register()
class BatchCompose():
    def __init__(self, to_tensor_keys=["imgs", "masks", "labels"]):
        self.to_tensor_keys = to_tensor_keys

    def __call__(self, batch):
        result_batch = []
        for index in range(len(batch)):
            data = {}
            for key, value in batch[index].items():
                if key in self.to_tensor_keys:
                    if not torch.is_tensor(value):
                        data[key] = torch.tensor(value)
                    else:
                        data[key] = value
                else:
                    data[key] = value
            result_batch.append(data)
        return result_batch

@PIPLINE.register()
class RawFramePipeline():
    def __init__(self,
                 decode=None,
                 sample=None,
                 transform=None):
        self.decode = VideoDecoder(**decode)
        self.sample = VideoStreamSampler(**sample)
        self.transform = VideoStreamTransform(transform)

    def __call__(self, results):
        # decode
        results = self.decode(results)
        # sample
        results = self.sample(results)
        # transform
        results = self.transform(results)
        return results

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

class VideoFrameSample():
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

class VideoStreamSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 sample_rate=4,
                 clip_seg_num=15,
                 sliding_window=60,
                 ignore_index=-100,
                 channel_mode="RGB",
                 sample_mode='random'
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.sample = VideoFrameSample(mode = sample_mode)
    
    def _all_valid_frames(self, start_frame, end_frame, video_len, container, labels):
        imgs = []
        vid_end_frame = end_frame
        if end_frame > video_len:
            vid_end_frame = video_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels = self._labels_sample(labels, samples_idx=frames_idx).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        if len(imgs) < self.clip_seg_num:
            np_frames = np_frames[-1].asnumpy().copy()
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
                
        mask = np.ones((labels.shape[0]))

        return imgs, labels, mask
    
    def _some_valid_frames(self, start_frame, end_frame, video_len, frames_len, container, labels):
        imgs = []
        frames_idx = self.sample(start_frame, video_len, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=frames_len).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))
        np_frames = np.zeros_like(np_frames[0])
        pad_len = self.clip_seg_num - len(imgs)
        for i in range(pad_len):
            imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])

        return imgs, labels, mask
    
    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[0]):
        if self.is_train:
            sample_labels = labels[samples_idx]
            sample_labels = np.repeat(sample_labels, repeats=self.sample_rate, axis=-1)
        else:
            sample_labels = labels[start_frame:end_frame]
        return sample_labels

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        video_len = int(results['video_len'])
        results['frames_len'] = frames_len
        container = results['frames']
        labels = results['raw_labels']

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len and end_frame < frames_len:
            imgs, labels, mask = self._all_valid_frames(start_frame, end_frame, video_len, container, labels)
        elif start_frame < frames_len and end_frame >= frames_len:
            imgs, labels, mask = self._some_valid_frames(start_frame, end_frame, video_len, frames_len, container, labels)
        else:
            imgs = []
            np_frames = np.zeros((240, 320, 3))
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        return results

class VideoStreamTransform():
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
        imgs = []
        for img in results['imgs']:
            img = self.imgs_transforms_pipeline(img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        return results