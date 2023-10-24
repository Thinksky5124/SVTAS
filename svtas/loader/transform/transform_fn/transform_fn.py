'''
Author       : Thyssen Wen
Date         : 2022-10-24 20:17:17
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-24 19:47:53
Description  : Transform Class Function
FilePath     : /SVTAS/svtas/loader/transform/transform_fn/transform_fn.py
'''
import abc
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Any, Iterable, Dict
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
    "DtypeToUInt8",
    "SegmentationLabelsToBoundaryProbability",
    "LabelsToOneHot",
    "NumpyExpandims",
    "TensorExpandims",
    "NumpyDataTypeTransform",
    "DropResultsByKeyName",
    "RenameResultTransform",
    "TensorToNumpy"
]

class BaseTransformFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass

class DtypeToUInt8(BaseTransformFunction):
    def __call__(self, flow_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # dtype -> uint8
        return flow_tensor.to(torch.uint8)

class ToUInt8(BaseTransformFunction):
    def __init__(self, bound=20):
        self.bound = bound

    def __call__(self, flow_tensor: torch.FloatTensor) -> torch.FloatTensor:
        # preprocessing as in
        # https://github.com/deepmind/kinetics-i3d/issues/61#issuecomment-506727158
        # but for pytorch
        # [-bound, bound] -> [0, 255]
        flow_tensor = (flow_tensor + self.bound) * (255.0 / (2 * self.bound))
        return flow_tensor.round().to(torch.uint8)

class XToTensor(BaseTransformFunction):

    def __call__(self, feature):
        return torch.tensor(feature)
        
class ToFloat(BaseTransformFunction):

    def __call__(self, byte_img):
        return byte_img.float()

class NormalizeColorTo1(BaseTransformFunction):
    def __call__(self, img):
        return img / 255.0

class ScaleTo1_1(BaseTransformFunction):
    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return (2 * tensor / 255) - 1


class Clamp(BaseTransformFunction):

    def __init__(self, min_val, max_val) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, tensor):
        return torch.clamp(tensor, min=self.min_val, max=self.max_val)

class PermuteAndUnsqueeze(BaseTransformFunction):

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return tensor.permute(1, 0, 2, 3).unsqueeze(0)

class TensorCenterCrop(BaseTransformFunction):

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

class ResizeImproved(BaseTransformFunction):

    def __init__(self, size: int, resize_to_smaller_edge: bool = True, interpolation=Image.BILINEAR):
        self.size = size
        self.resize_to_smaller_edge = resize_to_smaller_edge
        self.interpolation = interpolation

    def __call__(self, img):
        return resize(img, self.size, self.resize_to_smaller_edge, self.interpolation)

class TensorImageResize(BaseTransformFunction):
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

class TensorPermute(BaseTransformFunction):
    def __init__(self, permute_list=None):
        self.permute_list = permute_list

    def __call__(self, tensor: torch.FloatTensor) -> torch.FloatTensor:
        return tensor.permute(self.permute_list)

class OpencvToPIL(BaseTransformFunction):
    def __init__(self, in_channel_model="BGR"):
        self.in_channel_model = in_channel_model

    def __call__(self, img: np.ndarray) -> Image:
        if self.in_channel_model == "RGB":
            img = Image.fromarray(img)
        else:
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return img

class RandomShortSideScaleJitter(BaseTransformFunction):
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

class LabelsToOneHot(BaseTransformFunction):
    def __init__(self,
                 num_classes: int,
                 sample_rate = 1,
                 ignore_index: int = -100) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
    
    def __call__(self, labels: torch.LongTensor) -> Any:
        # labels [T]
        # deal label over num_classes
        labels = labels[::self.sample_rate]
        # [T]
        y = torch.zeros(labels.shape, dtype=labels.dtype)
        refine_label = torch.where(labels != self.ignore_index, labels, y)
        # [T C]
        ce_y = F.one_hot(refine_label, num_classes=self.num_classes)

        raw_labels_repeat = torch.tile(labels.unsqueeze(1), dims=[1, self.num_classes])
        ce_y = torch.where(raw_labels_repeat != self.ignore_index, ce_y, torch.zeros(ce_y.shape, dtype=ce_y.dtype)).float()
        return ce_y

class SegmentationLabelsToBoundaryProbability(BaseTransformFunction):
    def __init__(self,
                 smooth_method = 'guassion',
                 sigma = 1,
                 sample_rate = 1,
                 need_norm = False) -> None:
        super().__init__()
        assert smooth_method in ['guassion', 'none']
        self.smooth_method = smooth_method
        self.sigma = sigma
        self.sample_rate = sample_rate
        self.need_norm = need_norm
        self.kernel_size = int(2 * 4 * sigma + 1)
        if smooth_method == 'guassion':
            self.kernel, self.max = self.create_1d_gaussian_kernel(sigma=sigma)
    
    def create_1d_gaussian_kernel(self, sigma, truncate: float = 4.0, order=0):
        sd = float(sigma)
        # make the radius of the filter equal to truncate standard deviations
        lw = int(truncate * sd + 0.5)

        kernel = self._gaussian_kernel1d(sigma, order, lw)[::-1]
        kernel = torch.from_numpy(np.ascontiguousarray(kernel)).float()
        kernel = kernel.view(1, 1, self.kernel_size)

        temp_seq = torch.zeros((1, 1, 2 * self.kernel_size))
        temp_seq[0, 0, temp_seq.shape[-1] // 2] = 1
        temp_seq[0, 0, temp_seq.shape[-1] // 2 - 1] = 1
        temp_seq = torch.nn.ReflectionPad1d((self.kernel_size - 1) // 2)(temp_seq)
        norm_max = F.conv1d(temp_seq, kernel).max()
        
        return kernel, norm_max

    def _gaussian_kernel1d(self, sigma, order, radius):
        """
        Computes a 1-D Gaussian convolution kernel.
        """
        if order < 0:
            raise ValueError('order must be non-negative')
        exponent_range = np.arange(order + 1)
        sigma2 = sigma * sigma
        x = np.arange(-radius, radius+1)
        phi_x = np.exp(-0.5 / sigma2 * x ** 2)
        phi_x = phi_x / phi_x.sum()

        if order == 0:
            return phi_x
        else:
            # f(x) = q(x) * phi(x) = q(x) * exp(p(x))
            # f'(x) = (q'(x) + q(x) * p'(x)) * phi(x)
            # p'(x) = -1 / sigma ** 2
            # Implement q'(x) + q(x) * p'(x) as a matrix operator and apply to the
            # coefficients of q(x)
            q = np.zeros(order + 1)
            q[0] = 1
            D = np.diag(exponent_range[1:], 1)  # D @ q(x) = q'(x)
            P = np.diag(np.ones(order)/-sigma2, -1)  # P @ q(x) = q(x) * p'(x)
            Q_deriv = D + P
            for _ in range(order):
                q = Q_deriv.dot(q)
            q = (x[:, None] ** exponent_range).dot(q)
            return q * phi_x
    
    def __call__(self, label) -> Any:
        # labels [T]
        label = label[::self.sample_rate]
        # define the frame where new action starts as boundary frame
        boundary = torch.zeros_like(label, dtype=torch.float)
        last = label[0]
        boundary[0] = 1
        for i in range(1, label.shape[0]):
            if last != label[i]:
                boundary[i] = 1
                last = label[i]

        boundary = boundary.view(1, 1, -1)
        if self.smooth_method != 'none':
            boundary = torch.nn.ReflectionPad1d((self.kernel_size - 1) // 2)(boundary)
            boundary = F.conv1d(boundary, self.kernel)
            if self.need_norm:
                boundary[boundary > self.max] = self.max
                boundary /= boundary.max()

        return boundary[0, 0, :]

class NumpyExpandims(BaseTransformFunction):
    def __init__(self, axis = 0) -> None:
        super().__init__()
        self.axis = axis

    def __call__(self, x : np.array) -> np.array:
        return np.expand_dims(x, axis=self.axis)
    
class TensorExpandims(BaseTransformFunction):
    def __init__(self, dim = 0) -> None:
        super().__init__()
        self.dim = dim
    
    def __call__(self, x : torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(self.dim)

class NumpyDataTypeTransform(BaseTransformFunction):
    def __init__(self, dtype = "float32") -> None:
        super().__init__()
        assert dtype in ['float32', 'int64']

        if dtype == "float32":
            self.dtype = np.float32
        elif dtype == "int64":
            self.dtype = np.int64
    
    def __call__(self, x : np.array) -> np.array:
        return x.astype(self.dtype)

class DropResultsByKeyName(BaseTransformFunction):
    def __init__(self,
                 drop_keys_list = ["frames"]) -> None:
        super().__init__()
        self.drop_keys_list = drop_keys_list
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        for key_name in self.drop_keys_list:
            if key_name in results:
                results.pop(key_name)
        return results

class RenameResultTransform(BaseTransformFunction):
    def __init__(self,
                 rename_pair_dict: Dict) -> None:
        super().__init__()
        self.rename_pair_dict = rename_pair_dict
    
    def __call__(self, results: Dict[str, Any]) -> Dict[str, Any]:
        for key, name in self.rename_pair_dict.items():
            if key in results:
                results[name] = results[key]
                results.pop(key)
        return results

class TensorToNumpy(BaseTransformFunction):

    def __call__(self, x: torch.Tensor):
        return x.numpy()