'''
Author       : Thyssen Wen
Date         : 2022-10-24 20:17:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 21:06:10
Description  : Transform Class Function
FilePath     : /SVTAS/svtas/loader/transform/transform_fn/transform_fn.py
'''
import math
import numpy as np
import cv2
import torch
from typing import Iterable
from PIL import Image

__all__ = [
    "XToTensor",
    "ToFloat",
    "ScaleTo1_1",
    "NormalizeColorTo1",
    "Clamp",
    "PermuteAndUnsqueeze",
    "TensorCenterCrop",
    "ResizeImproved",
    "ToUInt8",
    "TensorImageResize",
    "TensorPermute",
    "OpencvToPIL",
    "RandomShortSideScaleJitter",
    "DtypeToUInt8"
]

class DtypeToUInt8(object):
    def __call__(self, flow_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # dtype -> uint8
        return flow_tensor.to(torch.uint8)

class ToUInt8(object):
    def __init__(self, bound=20):
        self.bound = bound

    def __call__(self, flow_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # preprocessing as in
        # https://github.com/deepmind/kinetics-i3d/issues/61#issuecomment-506727158
        # but for pytorch
        # [-bound, bound] -> [0, 255]
        flow_tensor = (flow_tensor + self.bound) * (255.0 / (2 * self.bound))
        return flow_tensor.round().to(torch.uint8)

class XToTensor(object):

    def __call__(self, feature):
        return torch.tensor(feature)
        
class ToFloat(object):

    def __call__(self, byte_img):
        return byte_img.float()

class NormalizeColorTo1:
    def __call__(self, img):
        return img / 255.0

class ScaleTo1_1(object):
    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return (2 * tensor / 255) - 1


class Clamp(object):

    def __init__(self, min_val, max_val) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return torch.clamp(tensor, min=self.min_val, max=self.max_val)

class PermuteAndUnsqueeze(object):

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return tensor.permute(1, 0, 2, 3).unsqueeze(0)

class TensorCenterCrop(object):

    def __init__(self, crop_size: int) -> None:
        self.crop_size = crop_size

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        H, W = tensor.size(-2), tensor.size(-1)
        from_H = ((H - self.crop_size) // 2)
        from_W = ((W - self.crop_size) // 2)
        to_H = from_H + self.crop_size
        to_W = from_W + self.crop_size
        return tensor[..., from_H:to_H, from_W:to_W]

def resize(img, size, resize_to_smaller_edge=True, interpolation=Image.BILINEAR):
    r"""
    (v-iashin): this is almost the same implementation as in PyTorch except it has no _is_pil_image() check
    and has an extra argument governing what happens if `size` is `int`.

    Reference: https://pytorch.org/docs/1.6.0/_modules/torchvision/transforms/functional.html#resize
    Resize the input PIL Image to the given size.

    Args:
        img (PIL Image): Image to be resized.
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller (bigger depending on `resize_to_smaller_edge`) edge of the image will be matched
            to this number maintaining
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            :math:`\left(\text{size} \times \frac{\text{height}}{\text{width}}, \text{size}\right)`
        resize_to_smaller_edge (bool, optional): if True the smaller edge is matched to number in `size`,
            if False, the bigger edge is matched to it.
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``

    Returns:
        PIL Image: Resized image.
    """
    if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        w, h = img.size
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if (w < h) == resize_to_smaller_edge:
            ow = size
            oh = int(size * h / w)
            return img.resize((ow, oh), interpolation)
        else:
            oh = size
            ow = int(size * w / h)
            return img.resize((ow, oh), interpolation)
    else:
        return img.resize(size[::-1], interpolation)

class ResizeImproved(object):

    def __init__(self, size: int, resize_to_smaller_edge: bool = True, interpolation=Image.BILINEAR):
        self.size = size
        self.resize_to_smaller_edge = resize_to_smaller_edge
        self.interpolation = interpolation

    def __call__(self, img):
        return resize(img, self.size, self.resize_to_smaller_edge, self.interpolation)

class TensorImageResize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, vid):
        # NOTE: for those functions, which generally expect mini-batches, we keep them
        # as non-minibatch so that they are applied as if they were 4d (thus image).
        # this way, we only apply the transformation in the spatial domain
        interpolation = 'bilinear'
        # NOTE: using bilinear interpolation because we don't work on minibatches
        # at this level
        scale = None
        if isinstance(self.size, int):
            scale = float(self.size) / min(vid.shape[-2:])
            size = None
        else:
            size = self.size
        return torch.nn.functional.interpolate(
            vid.unsqueeze(0), size=size, scale_factor=scale, mode=interpolation, align_corners=False,
            recompute_scale_factor=False
        ).squeeze(0)

class TensorPermute(object):
    def __init__(self, permute_list=None):
        self.permute_list = permute_list

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return tensor.permute(self.permute_list)

class OpencvToPIL(object):
    def __init__(self, in_channel_model="BGR"):
        self.in_channel_model = in_channel_model

    def __call__(self, img: np.ndarray) -> Image:
        if self.in_channel_model == "RGB":
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

class RandomShortSideScaleJitter(object):
    def __init__(self, min_size, max_size, inverse_uniform_sampling=False):
        self.min_size = min_size
        self.max_size = max_size
        self.inverse_uniform_sampling = inverse_uniform_sampling

    def __call__(self, images: torch.tensor):
        """
        ref:https://github.com/facebookresearch/SlowFast/blob/main/slowfast/datasets/transform.py
        Perform a spatial short scale jittering on the given images and
        corresponding boxes.
        Args:
            images (tensor): images to perform scale jitter. Dimension is
                `num frames` x `channel` x `height` x `width`.
            min_size (int): the minimal size to scale the frames.
            max_size (int): the maximal size to scale the frames.
            inverse_uniform_sampling (bool): if True, sample uniformly in
                [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
                scale. If False, take a uniform sample from [min_scale, max_scale].
        Returns:
            (tensor): the scaled images with dimension of
                `num frames` x `channel` x `new height` x `new width`.
        """
        images = images.unsqueeze(0)
        if self.inverse_uniform_sampling:
            size = int(
                round(1.0 / np.random.uniform(1.0 / self.max_size, 1.0 / self.min_size))
            )
        else:
            size = int(round(np.random.uniform(self.min_size, self.max_size)))

        height = images.shape[2]
        width = images.shape[3]
        if (width <= height and width == size) or (
            height <= width and height == size
        ):
            return images
        new_width = size
        new_height = size
        if width < height:
            new_height = int(math.floor((float(height) / width) * size))
        else:
            new_width = int(math.floor((float(width) / height) * size))

        return torch.nn.functional.interpolate(
                    images,
                    size=(new_height, new_width),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
