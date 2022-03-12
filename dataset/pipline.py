import torchvision.transforms as transforms
import decord as de
import numpy as np
import copy
from PIL import Image

class BatchCompose(object):
    def __init__(self, clip_seg_num=15,sample_rate=4):
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate

    def __call__(self, batch):
        max_imgs_len = 0
        max_labels_len = 0
        for mini_batch in batch:
            if max_imgs_len < mini_batch[0].shape[0]:
                max_imgs_len = mini_batch[0].shape[0]
            if max_labels_len < mini_batch[1].shape[0]:
                max_labels_len = mini_batch[1].shape[0]

        max_imgs_len = max_imgs_len + (self.clip_seg_num - max_imgs_len % self.clip_seg_num)
        max_labels_len = max_labels_len + ((self.clip_seg_num * self.sample_rate) - max_labels_len % (self.clip_seg_num * self.sample_rate))

        # shape imgs and labels len
        for batch_id in range(len(batch)):
            mini_batch_list = []
            list(batch[batch_id])
            # imgs
            pad_imgs_len = max_imgs_len - batch[batch_id][0].shape[0]
            pad_imgs = np.zeros([pad_imgs_len] + list(batch[batch_id][0].shape[1:]), dtype=batch[batch_id][0].dtype)
            # pad_imgs = np.random.normal(size = [pad_imgs_len] + list(batch[batch_id][0].shape[1:])).astype(batch[batch_id][0].dtype)
            mini_batch_list.append(np.concatenate([batch[batch_id][0], pad_imgs], axis=0))
            # lables
            pad_labels_len = max_labels_len - batch[batch_id][1].shape[0]
            pad_labels = np.full([pad_labels_len] + list(batch[batch_id][1].shape[1:]), -100, dtype=batch[batch_id][1].dtype)
            labels = np.concatenate([batch[batch_id][1], pad_labels], axis=0)
            mini_batch_list.append(labels)
            # masks
            mask = labels != -100
            mask = mask.astype(np.float32)
            mini_batch_list.append(mask)
            # vid
            mini_batch_list.append(batch[batch_id][-1])
            batch[batch_id] = tuple(mini_batch_list)
        result_batch = copy.deepcopy(batch)
        return result_batch

class Pipeline(object):
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

class VideoDecoder(object):
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
        frames_len = len(container)
        results['frames'] = container
        results['frames_len'] = frames_len
        
        return results

class VideoStreamSampler(object):
    """
    Sample frames id.
    NOTE: Use PIL to read image here, has diff with CV2
    Args:
        num_seg(int): number of segments.
        seg_len(int): number of sampled frames in each segment.
        valid_mode(bool): True or False.
        select_left: Whether to select the frame to the left in the middle when the sampling interval is even in the test mode.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 seg_len,
                 sample_rate=4,
                 clip_seg_num=15,
                 sliding_window=15,
                 frame_interval=None,
                 valid_mode=False,
                 select_left=False,
                 dense_sample=False,
                 linspace_sample=False
                 ):
        self.sample_rate = sample_rate
        self.seg_len = seg_len
        self.frame_interval = frame_interval
        self.valid_mode = valid_mode
        self.select_left = select_left
        self.dense_sample = dense_sample
        self.linspace_sample = linspace_sample
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window

    def __call__(self, results):
        """
        Args:
            frames_len: length of frames.
        return:
            sampling id.
        """
        frames_len = int(results['frames_len'])
        results['frames_len'] = frames_len
        container = results['frames']
        imgs = []
        labels = results['labels']

        # generate sample index
        start_frame = results['sample_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len and end_frame < frames_len:
            frames_idx = list(range(start_frame, end_frame, self.sample_rate))
            frames_select = container.get_batch(frames_idx)
            # dearray_to_img
            np_frames = frames_select.asnumpy()
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i]
                imgs.append(Image.fromarray(imgbuf, mode='RGB'))
            mask = np.ones((np_frames.shape[0]))
        elif start_frame < frames_len and end_frame > frames_len:
            frames_idx = list(range(start_frame, frames_len, self.sample_rate))
            frames_select = container.get_batch(frames_idx)
            # dearray_to_img
            np_frames = frames_select.asnumpy()
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i]
                imgs.append(Image.fromarray(imgbuf, mode='RGB'))
            vaild_mask = np.ones((np_frames.shape[0]))
            np_frames = np.zeros_like(np_frames[0])
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode='RGB'))
            void_mask = np.zeros((pad_len))
            mask = np.concatenate([vaild_mask, void_mask], axis=0)
        else:
            # ! find shape
            np_frames = np.zeros_like(np_frames[0])
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode='RGB'))
            mask = np.zeros((pad_len))

        results['imgs'] = imgs
        results['labels'] = labels
        results['mask'] = mask
        return results

class VideoStreamTransform(object):
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
        imgs = self.imgs_transforms_pipeline(results['imgs'])
        results['imgs'] = copy.deepcopy(imgs)
        return results