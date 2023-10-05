'''
Author       : Thyssen Wen
Date         : 2023-02-08 11:01:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 15:22:25
Description  : ASRF Postprocessing ref:https://github.com/torchtorch/torchVideo/blob/develop/torchvideo/modeling/framework/segmenters/utils.py
FilePath     : /SVTAS/svtas/model/post_precessings/refine_method/asrf.py
'''
# https://github.com/yiskw713/asrf/libs/postprocess.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class ASRFRefineMethod:
    """
    ASRF post processing is to refine action boundary
    Args:
        outputs_cls: the results of action segmentation. (N, T) or (N, C, T)
        outputs_boundary: action boundary probability. (N, 1, T)
        refinement_method: the way of refine predict boundary and classification. str
        boundary_threshold: the threshold of the size of action segments. float(default=0.7)
        theta_t: the threshold of the size of action segments. int(default=15)
        kernel_size: Size of the gaussian kernel. int(default=15)
    Return:
        preds output. (N, T)
    """
    def __init__(self,
                refinement_method="refinement_with_boundary",
                boundary_threshold=0.7,
                theta_t=15,
                kernel_size=15):
        func = [
            "refinement_with_boundary",
            "relabeling",
            "smoothing",
        ]

        assert refinement_method in func

        self.boundary_threshold = boundary_threshold
        self.theta_t = theta_t
        self.kernel_size = kernel_size
        if refinement_method == "smoothing":
            self.refine_func = self.gaussian_smoothing
        elif refinement_method == "relabeling":
            self.refine_func = self.relabeling
        elif refinement_method == "refinement_with_boundary":
            self.refine_func = self.refinement_with_boundary
        else:
            self.refine_func = self.default_func
        
    def gaussian_smoothing(self, outputs_cls, outputs_boundary):
        filter_func = GaussianSmoothing(self.kernel_size)
        preds = smoothing(outputs_cls, filter_func)
        return preds
    
    def relabeling(self, outputs_cls, outputs_boundary):
        return relabeling(outputs_cls, self.theta_t)
    
    def refinement_with_boundary(self, outputs_cls, outputs_boundary):
        return refinement_with_boundary(outputs_cls, outputs_boundary,
                                            self.boundary_threshold)

    def default_func(self, outputs_cls, outputs_boundary):
        preds = np.zeros((1, 1))
        return preds

    def __call__(self, outputs_cls, outputs_boundary) -> torch.Tensor:

        return torch.tensor(self.refine_func(outputs_cls, outputs_boundary))

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a 1d tensor.
    Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
    """

    def __init__(self, kernel_size=15, sigma=1.0):
        super().__init__()
        self.kernel_size = kernel_size

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrid = torch.arange(kernel_size)

        meshgrid = torch.cast(meshgrid, dtype='float32')

        mean = (kernel_size - 1) / 2
        kernel = kernel / (sigma * math.sqrt(2 * math.pi))
        kernel = kernel * torch.exp(-(((meshgrid - mean) / sigma)**2) / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        # kernel = kernel / torch.max(kernel)

        self.kernel = torch.reshape(kernel, [1, 1, -1])

    def forward(self, inputs):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        _, c, _ = inputs.shape
        inputs = F.pad(inputs,
                       pad=((self.kernel_size - 1) // 2,
                            (self.kernel_size - 1) // 2),
                       mode="reflect",
                       data_format='NCL')

        kernel = torch.expand(self.kernel, shape=[c, 1, self.kernel_size])
        return F.conv1d(inputs, weight=kernel, groups=c)


def argrelmax(prob, threshold=0.7):
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold
    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=np.bool),
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]),
            np.zeros((1), dtype=np.bool),
        ],
        axis=0,
    )

    peak_idx = np.where(peak)[0].tolist()

    return peak_idx


def is_probability(x):
    assert x.ndim == 3

    if x.shape[1] == 1:
        # sigmoid
        if x.min() >= 0 and x.max() <= 1:
            return True
        else:
            return False
    else:
        # softmax
        _sum = np.sum(x, axis=1).astype(np.float32)
        _ones = np.ones_like(_sum, dtype=np.float32)
        return np.allclose(_sum, _ones)


def convert2probability(x):
    """
    Args: x (N, C, T)
    """
    assert x.ndim == 3

    if is_probability(x):
        return x
    else:
        if x.shape[1] == 1:
            # sigmoid
            prob = 1 / (1 + np.exp(-x))
        else:
            # softmax
            prob = np.exp(x) / np.sum(np.exp(x), axis=1)
        return prob.astype(np.float32)


def convert2label(x):
    assert x.ndim == 2 or x.ndim == 3

    if x.ndim == 2:
        return x.astype(np.int64)
    else:
        if not is_probability(x):
            x = convert2probability(x)

        label = np.argmax(x, axis=1)
        return label.astype(np.int64)


def refinement_with_boundary(outputs, boundaries, boundary_threshold):
    """
    Get segments which is defined as the span b/w two boundaries,
    and decide their classes by majority vote.
    Args:
        outputs: numpy array. shape (N, C, T)
            the model output for frame-level class prediction.
        boundaries: numpy array.  shape (N, 1, T)
            boundary prediction.
        boundary_threshold: the threshold of the size of action segments. float(default=0.7)
    Return:
        preds: np.array. shape (N, T)
            final class prediction considering boundaries.
    """

    preds = convert2label(outputs)
    boundaries = convert2probability(boundaries)

    for i, (output, pred, boundary) in enumerate(zip(outputs, preds,
                                                     boundaries)):
        idx = argrelmax(boundary[0, :], threshold=boundary_threshold)

        # add the index of the last action ending
        T = pred.shape[0]
        idx.append(T)

        # majority vote
        for j in range(len(idx) - 1):
            count = np.bincount(pred[idx[j]:idx[j + 1]])
            modes = np.where(count == count.max())[0]
            if len(modes) == 1:
                mode = modes
            else:
                if outputs.ndim == 3:
                    # if more than one majority class exist
                    prob_sum_max = 0
                    for m in modes:
                        prob_sum = output[m, idx[j]:idx[j + 1]].sum()
                        if prob_sum_max < prob_sum:
                            mode = m
                            prob_sum_max = prob_sum
                else:
                    # decide first mode when more than one majority class
                    # have the same number during oracle experiment
                    mode = modes[0]

            preds[i, idx[j]:idx[j + 1]] = mode
    return preds


def relabeling(outputs, theta_t):
    """
        Relabeling small action segments with their previous action segment
        Args:
            output: the results of action segmentation. (N, T) or (N, C, T)
            theta_t: the threshold of the size of action segments.
        Return:
            relabeled output. (N, T)
        """

    preds = convert2label(outputs)

    for i in range(preds.shape[0]):
        # shape (T,)
        last = preds[i][0]
        cnt = 1
        for j in range(1, preds.shape[1]):
            if last == preds[i][j]:
                cnt += 1
            else:
                if cnt > theta_t:
                    cnt = 1
                    last = preds[i][j]
                else:
                    preds[i][j - cnt:j] = preds[i][j - cnt - 1]
                    cnt = 1
                    last = preds[i][j]

        if cnt <= theta_t:
            preds[i][j - cnt:j] = preds[i][j - cnt - 1]

    return preds


def smoothing(outputs, filter_func):
    """
        Smoothing action probabilities with gaussian filter.
        Args:
            outputs: frame-wise action probabilities. (N, C, T)
        Return:
            predictions: final prediction. (N, T)
        """

    outputs = convert2probability(outputs)
    outputs = filter_func(torch.tensor(outputs)).numpy()

    preds = convert2label(outputs)
    return preds


def _calculate_fan_in_and_fan_out(tensor):
    dimensions = len(tensor.shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed \
        for tensor with fewer than 2 dimensions")

    if dimensions == 2:  # Linear
        fan_in = tensor.shape[1]
        fan_out = tensor.shape[0]
    else:
        num_input_fmaps = tensor.shape[1]
        num_output_fmaps = tensor.shape[0]
        receptive_field_size = 1
        if tensor.dim() > 2:
            receptive_field_size = tensor[0][0].numel()
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def calculate_gain(nonlinearity=None, a=None):
    if nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if a is not None:
            return math.sqrt(2.0 / (1 + a**2))
        else:
            return math.sqrt(2.0 / (1 + 0.01**2))
    elif nonlinearity == 'selu':
        return 3.0 / 4
    else:
        return 1


def KaimingUniform_like_torch(weight_npy,
                              mode='fan_in',
                              nonlinearity='leaky_relu'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    if mode == 'fan_in':
        fan_mode = fan_in
    else:
        fan_mode = fan_out
    a = math.sqrt(5.0)
    gain = calculate_gain(nonlinearity=nonlinearity, a=a)
    std = gain / math.sqrt(fan_mode)
    bound = math.sqrt(3.0) * std
    return np.random.uniform(-bound, bound, weight_npy.shape)


def init_bias(weight_npy, bias_npy):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(weight_npy)
    bound = 1.0 / math.sqrt(fan_in)
    return np.random.uniform(-bound, bound, bias_npy.shape)