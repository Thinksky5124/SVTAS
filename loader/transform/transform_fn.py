'''
Author       : Thyssen Wen
Date         : 2022-10-24 20:17:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 15:06:23
Description  : Transform Class Function
FilePath     : /SVTAS/loader/transform/transform_fn.py
'''
import torch
from typing import Iterable
import numpy as np
import cv2
from PIL import Image
import random
import mmcv

__all__ = [
    "FeatureToTensor",
    "ToFloat",
    "ScaleTo1_1",
    "Clamp",
    "PermuteAndUnsqueeze",
    "TensorCenterCrop",
    "ResizeImproved"
]

class FeatureToTensor(object):

    def __call__(self, feature):
        return torch.tensor(feature)
        
class ToFloat(object):

    def __call__(self, byte_img):
        return byte_img.float()
        
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
