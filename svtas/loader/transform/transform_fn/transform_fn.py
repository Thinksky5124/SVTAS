'''
Author       : Thyssen Wen
Date         : 2022-10-24 20:17:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 20:55:52
Description  : Transform Class Function
FilePath     : /SVTAS/svtas/loader/transform/transform_fn.py
'''
import torch
from typing import Iterable
from PIL import Image

__all__ = [
    "FeatureToTensor",
    "ToFloat",
    "ScaleTo1_1",
    "NormalizeColorTo1",
    "Clamp",
    "PermuteAndUnsqueeze",
    "TensorCenterCrop",
    "ResizeImproved",
    "ToUInt8",
    "TensorImageResize",
    "TensorPermute"
]

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

class FeatureToTensor(object):

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

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""
This implementation is based on
https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/mixup.py,
published under an Apache License 2.0.
COMMENT FROM ORIGINAL:
Mixup and Cutmix
Papers:
mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features (https://arxiv.org/abs/1905.04899) # NOQA
Code Reference:
CutMix: https://github.com/clovaai/CutMix-PyTorch
Hacked together by / Copyright 2020 Ross Wightman
"""

import numpy as np
import torch


def convert_to_one_hot(targets, num_classes, on_value=1.0, off_value=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        on_value (float): Target Value for ground truth class.
        off_value (float): Target Value for other classes.This value is used for
            label smoothing.
    """

    targets = targets.long().view(-1, 1)
    return torch.full(
        (targets.size()[0], num_classes), off_value, device=targets.device
    ).scatter_(1, targets, on_value)


def mixup_target(target, num_classes, lam=1.0, smoothing=0.0):
    """
    This function converts target class indices to one-hot vectors, given the
    number of classes.
    Args:
        targets (loader): Class labels.
        num_classes (int): Total number of classes.
        lam (float): lamba value for mixup/cutmix.
        smoothing (float): Label smoothing value.
    """
    off_value = smoothing / num_classes
    on_value = 1.0 - smoothing + off_value
    target1 = convert_to_one_hot(
        target,
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    target2 = convert_to_one_hot(
        target.flip(0),
        num_classes,
        on_value=on_value,
        off_value=off_value,
    )
    return target1 * lam + target2 * (1.0 - lam)


def rand_bbox(img_shape, lam, margin=0.0, count=None):
    """
    Generates a random square bbox based on lambda value.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        margin (float): Percentage of bbox dimension to enforce as margin (reduce amount of box outside image)
        count (int): Number of bbox to generate
    """
    ratio = np.sqrt(1 - lam)
    img_h, img_w = img_shape[-2:]
    cut_h, cut_w = int(img_h * ratio), int(img_w * ratio)
    margin_y, margin_x = int(margin * cut_h), int(margin * cut_w)
    cy = np.random.randint(0 + margin_y, img_h - margin_y, size=count)
    cx = np.random.randint(0 + margin_x, img_w - margin_x, size=count)
    yl = np.clip(cy - cut_h // 2, 0, img_h)
    yh = np.clip(cy + cut_h // 2, 0, img_h)
    xl = np.clip(cx - cut_w // 2, 0, img_w)
    xh = np.clip(cx + cut_w // 2, 0, img_w)
    return yl, yh, xl, xh


def get_cutmix_bbox(img_shape, lam, correct_lam=True, count=None):
    """
    Generates the box coordinates for cutmix.
    Args:
        img_shape (tuple): Image shape as tuple
        lam (float): Cutmix lambda value
        correct_lam (bool): Apply lambda correction when cutmix bbox clipped by
            image borders.
        count (int): Number of bbox to generate
    """

    yl, yu, xl, xu = rand_bbox(img_shape, lam, count=count)
    if correct_lam:
        bbox_area = (yu - yl) * (xu - xl)
        lam = 1.0 - bbox_area / float(img_shape[-2] * img_shape[-1])
    return (yl, yu, xl, xu), lam


class MixUp:
    """
    Apply mixup and/or cutmix for videos at batch level.
    mixup: Beyond Empirical Risk Minimization (https://arxiv.org/abs/1710.09412)
    CutMix: Regularization Strategy to Train Strong Classifiers with Localizable
        Features (https://arxiv.org/abs/1905.04899)
    """

    def __init__(
        self,
        mixup_alpha=1.0,
        cutmix_alpha=0.0,
        mix_prob=1.0,
        switch_prob=0.5,
        correct_lam=True,
        label_smoothing=0.1,
        num_classes=1000,
    ):
        """
        Args:
            mixup_alpha (float): Mixup alpha value.
            cutmix_alpha (float): Cutmix alpha value.
            mix_prob (float): Probability of applying mixup or cutmix.
            switch_prob (float): Probability of switching to cutmix instead of
                mixup when both are active.
            correct_lam (bool): Apply lambda correction when cutmix bbox
                clipped by image borders.
            label_smoothing (float): Apply label smoothing to the mixed target
                tensor. If label_smoothing is not used, set it to 0.
            num_classes (int): Number of classes for target.
        """
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mix_prob = mix_prob
        self.switch_prob = switch_prob
        self.label_smoothing = label_smoothing
        self.num_classes = num_classes
        self.correct_lam = correct_lam

    def _get_mixup_params(self):
        lam = 1.0
        use_cutmix = False
        if np.random.rand() < self.mix_prob:
            if self.mixup_alpha > 0.0 and self.cutmix_alpha > 0.0:
                use_cutmix = np.random.rand() < self.switch_prob
                lam_mix = (
                    np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
                    if use_cutmix
                    else np.random.beta(self.mixup_alpha, self.mixup_alpha)
                )
            elif self.mixup_alpha > 0.0:
                lam_mix = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            elif self.cutmix_alpha > 0.0:
                use_cutmix = True
                lam_mix = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
            lam = float(lam_mix)
        return lam, use_cutmix

    def _mix_batch(self, x):
        lam, use_cutmix = self._get_mixup_params()
        if lam == 1.0:
            return 1.0
        if use_cutmix:
            (yl, yh, xl, xh), lam = get_cutmix_bbox(
                x.shape,
                lam,
                correct_lam=self.correct_lam,
            )
            x[..., yl:yh, xl:xh] = x.flip(0)[..., yl:yh, xl:xh]
        else:
            x_flipped = x.flip(0).mul_(1.0 - lam)
            x.mul_(lam).add_(x_flipped)
        return lam

    def __call__(self, x, target):
        if self.mix_prob > 0.0:
            assert len(x) > 1, "Batch size should be greater than 1 for mixup."
            lam = self._mix_batch(x)
        elif self.mix_prob == 0.0:
            lam = 1.0
        else:
            raise NotImplementedError
        target = mixup_target(
            target, self.num_classes, lam, self.label_smoothing
        )
        return x, target