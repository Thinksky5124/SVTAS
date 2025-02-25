'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:32:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-18 22:41:25
Description  : Raw frame sampler
FilePath     : /ETESVS/loader/sampler/frame_sampler.py
'''
import numpy as np
import random
from PIL import Image
import cv2 
import albumentations as A
from ..builder import SAMPLER

class VideoFrameSample():
    def __init__(self, mode='random'):
        assert mode in ['random', 'uniform', 'linspace', 'random_choice'], 'not support mode'
        self.mode = mode
    
    def random_sample(self, start_idx, end_idx, sample_rate):
        sample_idx = list(
                random.sample(list(range(start_idx, end_idx)),
                    len(list(range(start_idx, end_idx, sample_rate)))))
        sample_idx.sort()
        return sample_idx

    def uniform_sample(self, start_idx, end_idx, sample_rate):
        return list(range(start_idx, end_idx, sample_rate))
    
    def linspace_sample(self, start_idx, end_idx, sample_num):
        return list(np.ceil(np.linspace(start_idx, end_idx, num=sample_num)).astype(np.int64))
    
    def random_choice_sample(self, start_idx, end_idx, sample_num):
        return list(random.sample(range(start_idx, end_idx), sample_num))

    def __call__(self, start_idx, end_idx, sample_rate):
        if self.mode == 'random':
            return self.random_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'uniform':
            return self.uniform_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'linspace':
            return self.linspace_sample(start_idx, end_idx, sample_rate)
        elif self.mode == 'random_choice':
            return self.random_choice_sample(start_idx, end_idx, sample_rate)
        else:
            raise NotImplementedError

@SAMPLER.register()
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
                 sample_mode='random',
                 aug=[]
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.sample = VideoFrameSample(mode = sample_mode)
        self.transform = A.ReplayCompose([
                    A.RandomBrightnessContrast(p=0.6),
                    A.HueSaturationValue(hue_shift_limit=(-10, 10), sat_shift_limit=(-10, 10), val_shift_limit=(-10, 10), p=1),
                    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.01, rotate_limit=4, p=1),
                    A.ImageCompression(quality_lower=75, quality_upper=95, p=1),
                ])
    def _all_valid_frames(self, start_frame, end_frame, video_len, container, labels):
        imgs = []
        vid_end_frame = end_frame
        if end_frame > video_len:
            vid_end_frame = video_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()

        ### augmentations ### 
        first_frame = np_frames[0]
        replayed = self.transform(image=first_frame)
        replay_data = replayed['replay']  
        augmented_video = np.empty_like(np_frames)
        for i in range(np_frames.shape[0]):
            augmented_frame = A.ReplayCompose.replay(replay_data, image=np_frames[i])['image']
            augmented_video[i] = augmented_frame
        np_frames = augmented_video
        import pdb; pdb.set_trace()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        if len(imgs) < self.clip_seg_num:
            np_frames = np_frames[-1].copy()
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
                
        mask = np.ones((labels.shape[0]))

        return imgs, labels, mask
    
    def _some_valid_frames(self, start_frame, end_frame, video_len, frames_len, container, labels):
        imgs = []
        small_frames_video_len = min(frames_len, video_len)
        frames_idx = self.sample(start_frame, video_len, self.sample_rate)
        label_frames_idx = self.sample(start_frame, small_frames_video_len, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=small_frames_video_len, samples_idx=label_frames_idx).copy()
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        if np_frames.shape[0] > 0:
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i].copy()
                imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))
            np_frames = np.zeros_like(np_frames[0])
        else:
            np_frames = np.zeros((240, 320, 3))
        pad_len = self.clip_seg_num - len(imgs)
        for i in range(pad_len):
            imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])

        return imgs, labels, mask
    
    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[]):
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

        small_frames_video_len = min(frames_len, video_len)

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < small_frames_video_len and end_frame < small_frames_video_len:
            imgs, labels, mask = self._all_valid_frames(start_frame, end_frame, video_len, container, labels)
        elif start_frame < small_frames_video_len and end_frame >= small_frames_video_len:
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



@SAMPLER.register()
class VideoStreamSamplerMultiLabel():
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
                 sample_mode='random',
                 aug=[]
                 ):
        self.sample_rate = sample_rate
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.sliding_window = sliding_window
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.sample = VideoFrameSample(mode = sample_mode)
        self.transform = A.ReplayCompose([
                    A.RandomBrightnessContrast(p=0.6),
                    A.HueSaturationValue(hue_shift_limit=(-30, 30), sat_shift_limit=(-30, 30), val_shift_limit=(-30, 30), p=0.6),
                    A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.4, rotate_limit=15, p=0.6),
                    A.ImageCompression(quality_lower=75, quality_upper=95, p=0.6),
                ])
        self.visualize = False
        
    def save_video(self, np_frames: np.ndarray, output_path: str, fps: int = 30) -> None:
        """
        Saves a numpy array of shape (num_frames, height, width, channels) as a video file.
        Args:
            np_frames (np.ndarray): Array containing video frames.
            output_path (str): Path to save the output video file.
            fps (int, optional): Frames per second. Defaults to 30.
        """
        # Extract frame dimensions
        num_frames, height, width, channels = np_frames.shape
        assert channels == 3, "Frames should have 3 channels (RGB)."

        # Define the video codec and create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Write each frame to the video
        for frame in np_frames:
            # Convert RGB (numpy default) to BGR (OpenCV default)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame_bgr)

        # Release the video writer
        out.release()
        print(f'Video saved to {output_path}')

    def _all_valid_frames(self, start_frame, end_frame, video_len, container, labels, branch_labels):
        imgs = []
        vid_end_frame = end_frame
        if end_frame > video_len:
            vid_end_frame = video_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()
        branch_labels= self._labels_sample(branch_labels, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()
        
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()

        ### augmentations ### 
        first_frame = np_frames[0]
        replayed = self.transform(image=first_frame)
        replay_data = replayed['replay']  
        augmented_video = np.empty_like(np_frames)
        for i in range(np_frames.shape[0]):
            augmented_frame = A.ReplayCompose.replay(replay_data, image=np_frames[i])['image']
            augmented_video[i] = augmented_frame
        np_frames = augmented_video

        if self.visualize:
            self.save_video(np_frames=np_frames, output_path='test.mp4', fps=1)

        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        if len(imgs) < self.clip_seg_num:
            np_frames = np_frames[-1].copy()
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
                
        mask = np.ones((labels.shape[0]))

        return imgs, labels, mask, branch_labels
    
    def _some_valid_frames(self, start_frame, end_frame, video_len, frames_len, container, labels, branch_labels):
        imgs = []
        small_frames_video_len = min(frames_len, video_len)
        frames_idx = self.sample(start_frame, video_len, self.sample_rate)
        label_frames_idx = self.sample(start_frame, small_frames_video_len, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=small_frames_video_len, samples_idx=label_frames_idx).copy()
        branch_labels = self._labels_sample(branch_labels, start_frame=start_frame, end_frame=small_frames_video_len, samples_idx=label_frames_idx).copy()
        
        frames_select = container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = frames_select.asnumpy()
        if np_frames.shape[0] > 0:
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i].copy()
                imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))
            np_frames = np.zeros_like(np_frames[0])
        else:
            np_frames = np.zeros((240, 320, 3))
        pad_len = self.clip_seg_num - len(imgs)
        for i in range(pad_len):
            imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])
        branch_labels = np.concatenate([branch_labels, np.full((mask_pad_len), self.ignore_index)])

        return imgs, labels, mask, branch_labels
    
    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[]):
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
        branch_labels = results['raw_branch_labels']
        small_frames_video_len = min(frames_len, video_len)

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < small_frames_video_len and end_frame < small_frames_video_len:
            imgs, labels, mask, branch_labels = self._all_valid_frames(start_frame, end_frame, video_len, container, labels, branch_labels)
        elif start_frame < small_frames_video_len and end_frame >= small_frames_video_len:
            imgs, labels, mask, branch_labels = self._some_valid_frames(start_frame, end_frame, video_len, frames_len, container, labels, branch_labels)
        else:
            imgs = []
            np_frames = np.zeros((240, 320, 3))
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)
            branch_labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        results['branch_labels'] = branch_labels.copy()

        return results

@SAMPLER.register()
class RGBFlowVideoStreamSampler():
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
    
    def _all_valid_frames(self, start_frame, end_frame, video_len, rgb_container, flow_container, labels):
        imgs = []
        flows = []
        vid_end_frame = end_frame
        if end_frame > video_len:
            vid_end_frame = video_len
        frames_idx = self.sample(start_frame, vid_end_frame, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=end_frame, samples_idx=frames_idx).copy()
        rgb_frames_select = rgb_container.get_batch(frames_idx)
        flow_frames_select = flow_container.get_batch(frames_idx)
        # dearray_to_img
        np_frames = rgb_frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        np_frames = flow_frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            flows.append(Image.fromarray(imgbuf, mode=self.channel_mode))

        if len(imgs) < self.clip_seg_num:
            np_frames = np_frames[-1].asnumpy().copy()
            pad_len = self.clip_seg_num - len(imgs)
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
    
        if len(flows) < self.clip_seg_num:
            np_frames = np_frames[-1].asnumpy().copy()
            pad_len = self.clip_seg_num - len(flows)
            for i in range(pad_len):
                flows.append(Image.fromarray(np_frames, mode=self.channel_mode))
                
        mask = np.ones((labels.shape[0]))
        return imgs, flows, labels, mask
    
    def _some_valid_frames(self, start_frame, end_frame, video_len, frames_len, rgb_container, flow_container, labels):
        imgs = []
        flows = []

        rgb_frames_idx = self.sample(start_frame, video_len, self.sample_rate)
        flow_frames_idx = self.sample(start_frame, video_len - 1, self.sample_rate)
        label_frames_idx = self.sample(start_frame, frames_len, self.sample_rate)
        labels = self._labels_sample(labels, start_frame=start_frame, end_frame=frames_len, samples_idx=label_frames_idx).copy()
        rgb_frames_select = rgb_container.get_batch(rgb_frames_idx)
        flow_frames_select = flow_container.get_batch(flow_frames_idx)
        # dearray_to_img
        np_frames = rgb_frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            imgs.append(Image.fromarray(imgbuf, mode=self.channel_mode))
        np_frames = np.zeros_like(np_frames[0])
        pad_len = self.clip_seg_num - len(imgs)
        for i in range(pad_len):
            imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
        
        np_frames = flow_frames_select.asnumpy()
        for i in range(np_frames.shape[0]):
            imgbuf = np_frames[i].copy()
            flows.append(Image.fromarray(imgbuf, mode=self.channel_mode))
        np_frames = np.zeros_like(np_frames[0])
        pad_len = self.clip_seg_num - len(flows)
        for i in range(pad_len):
            flows.append(Image.fromarray(np_frames, mode=self.channel_mode))
            
        vaild_mask = np.ones((labels.shape[0]))
        mask_pad_len = self.clip_seg_num * self.sample_rate - labels.shape[0]
        void_mask = np.zeros((mask_pad_len))
        mask = np.concatenate([vaild_mask, void_mask], axis=0)
        labels = np.concatenate([labels, np.full((mask_pad_len), self.ignore_index)])
        
        return imgs, flows, labels, mask
    
    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[]):
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
        rgb_container = results['rgb_frames']
        flow_container = results['flow_frames']
        labels = results['raw_labels']

        # generate sample index
        start_frame = results['sample_sliding_idx'] * self.sliding_window
        end_frame = start_frame + self.clip_seg_num * self.sample_rate
        if start_frame < frames_len - 1 and end_frame < frames_len - 1:
            imgs, flows, labels, mask = self._all_valid_frames(start_frame, end_frame, video_len, rgb_container, flow_container, labels)
        elif start_frame < frames_len - 1 and end_frame >= frames_len - 1:
            imgs, flows, labels, mask = self._some_valid_frames(start_frame, end_frame, video_len, frames_len, rgb_container, flow_container, labels)
        else:
            imgs = []
            flows = []
            np_frames = np.zeros((224, 224, 3))
            pad_len = self.clip_seg_num
            for i in range(pad_len):
                imgs.append(Image.fromarray(np_frames, mode=self.channel_mode))
            for i in range(pad_len):
                flows.append(Image.fromarray(np_frames, mode=self.channel_mode))
            mask = np.zeros((self.clip_seg_num * self.sample_rate))
            labels = np.full((self.clip_seg_num * self.sample_rate), self.ignore_index)

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['flows'] = flows[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['mask'] = mask.copy()
        return results

@SAMPLER.register()
class VideoSampler():
    """
    Sample frames id.
    Returns:
        frames_idx: the index of sampled #frames.
    """

    def __init__(self,
                 is_train=False,
                 clip_seg_num=15,
                 ignore_index=-100,
                 channel_mode="RGB",
                 sample_mode='linspace'
                 ):
        self.is_train = is_train
        self.clip_seg_num = clip_seg_num
        self.ignore_index = ignore_index
        self.channel_mode = channel_mode
        self.sample = VideoFrameSample(mode = sample_mode)
        
    
    def _labels_sample(self, labels, start_frame=0, end_frame=0, samples_idx=[]):
        if self.is_train:
            sample_labels = labels[samples_idx]
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
        imgs = []
        frames_idx = self.sample(0, video_len, self.clip_seg_num)
        labels = self._labels_sample(labels, start_frame=0, end_frame=frames_len, samples_idx=frames_idx).copy()
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

        results['imgs'] = imgs[:self.clip_seg_num].copy()
        results['labels'] = labels.copy()
        results['masks'] = mask.copy()
        return results