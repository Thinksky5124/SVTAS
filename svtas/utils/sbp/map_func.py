'''
Author       : Thyssen Wen
Date         : 2023-02-11 15:37:00
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-13 11:17:01
Description  : file content
FilePath     : /SVTAS/svtas/utils/sbp/map_func.py
'''
import abc
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Dict, Tuple
from functools import reduce
from operator import mul

class MaskMappingFunctor(metaclass=abc.ABCMeta):
    """Mask Mapping Functor Abstract Class
    
    This class will support StochasticBackPropagation by generate corresponding mask for forawrd
    and stochastic backward propagation.

    All mapping function should inherite this class!

    This class offer two interface:
        - x_map_feature_fn(x_shape: torch.Tensor.shape,
                           device: torch.DeviceObjType,
                           sample_dims: List[int],
                           keep_ratio_list: List[int],
                           sample_index_list: List[List[int]],
                           raw_sample_shape: List[int])
        - feature_map_y_fn(x_shape: torch.Tensor.shape,
                           device: torch.DeviceObjType,
                           sample_dims: List[int],
                           keep_ratio_list: List[int],
                           sample_index_list: List[List[int]],
                           raw_sample_shape: List[int])
    """
    def __init__(self) -> None:
        pass

    def x_map_feature_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        raise NotImplementedError("This method should be overwrite!")

    def feature_map_y_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        raise NotImplementedError("This method should be overwrite!")

class SameDimensionMapFunctor(MaskMappingFunctor):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def same_dimension_map(x_shape: torch.Tensor.shape,
                           device: torch.DeviceObjType,
                           sample_dims: List[int],
                           keep_ratio_list: List[int],
                           sample_index_list: List[List[int]],
                           raw_sample_shape: List[int]):
        sample_shape = list(x_shape)

        for dim, keep_ratio in zip(sample_dims, keep_ratio_list):
            T = sample_shape[dim]
            d_T = 1 if T // keep_ratio <= 0 else T // keep_ratio
            sample_shape[dim] = d_T
        
        grad_mask_false = torch.zeros(x_shape, dtype=torch.bool, device=device)
        grad_mask_true = torch.ones_like(grad_mask_false)
        grad_mask = None
        match_sample_index_list = []

        for dim, sample_index in zip(sample_dims, sample_index_list):

            index = torch.floor(sample_index * (x_shape[dim] / raw_sample_shape[dim]))
            sample_sample_index = torch.arange(0, index.shape[0], index.shape[0] // sample_shape[dim])
            index = index[sample_sample_index].long()

            match_sample_index_list.append(index)

            index_dims = torch.ones(len(x_shape), dtype=torch.int32)
            index_dims[dim] = sample_shape[dim]
            index = index.reshape(index_dims.tolist())
            expand_dims = copy.deepcopy(list(x_shape))
            expand_dims[dim] = 1
            grad_mask_index = index.repeat(expand_dims).to(device=device)
            
            if grad_mask is None:
                grad_mask = torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)
            else:
                grad_mask = grad_mask * torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)

        grad_mask = grad_mask.bool()
        return grad_mask, match_sample_index_list, sample_shape

    def x_map_feature_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        return SameDimensionMapFunctor.same_dimension_map(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)
    
    def feature_map_y_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        return self.x_map_feature_fn(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)

class PermuteDimensionMapFunctor(MaskMappingFunctor):
    def __init__(self, permute_dims: List[int]) -> None:
        super().__init__()
        self.permute_dims = permute_dims
    
    @staticmethod
    def repermute_raw_shape_list(permute_dims: List[int], raw_sample_shape: List[int]):
        permute_raw_sample_shape = []
        for dst_idx in permute_dims:
            permute_raw_sample_shape.append(raw_sample_shape[dst_idx])
        return permute_raw_sample_shape
        
    @staticmethod
    def repermute_sample_info(permute_dims: List[int], sample_dims: List[int], keep_ratio_list: List[int], sample_index_list: List[List[int]],):
        permute_sample_dims = []
        permute_keep_ratio_list = []
        permute_sample_index_list = []
        for permute_idx, dim in enumerate(permute_dims):
            for key_index, sample_dim in enumerate(sample_dims):
                if dim == sample_dim:
                    permute_sample_dims.append(permute_idx)
                    permute_keep_ratio_list.append(keep_ratio_list[key_index])
                    permute_sample_index_list.append(sample_index_list[key_index])

        return permute_sample_dims, permute_keep_ratio_list, permute_sample_index_list

    def x_map_feature_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        raw_sample_shape = PermuteDimensionMapFunctor.repermute_raw_shape_list(self.permute_dims, raw_sample_shape)
        sample_dims, keep_ratio_list, sample_index_list = PermuteDimensionMapFunctor.repermute_sample_info(self.permute_dims, sample_dims,
                                                                                                           keep_ratio_list, sample_index_list)
        return SameDimensionMapFunctor.same_dimension_map(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)
    
    def feature_map_y_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]): 
        return self.x_map_feature_fn(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)

class Swin3DMLPMaskMappingFunctor(MaskMappingFunctor):
    def __init__(self, permute_dims: List[int]) -> None:
        super().__init__()
        self.permute_dims = permute_dims
    
    def x_map_feature_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        raw_sample_shape = PermuteDimensionMapFunctor.repermute_raw_shape_list(self.permute_dims, raw_sample_shape)
        sample_dims, keep_ratio_list, sample_index_list = PermuteDimensionMapFunctor.repermute_sample_info(self.permute_dims, sample_dims,
                                                                                                           keep_ratio_list, sample_index_list)
        return SameDimensionMapFunctor.same_dimension_map(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)
    
    def feature_map_y_fn(self,
                         x_shape: torch.Tensor.shape,
                         device: torch.DeviceObjType,
                         sample_dims: List[int],
                         keep_ratio_list: List[int],
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int]):
        sample_dims, keep_ratio_list, sample_index_list = PermuteDimensionMapFunctor.repermute_sample_info(self.permute_dims, sample_dims,
                                                                                                           keep_ratio_list, sample_index_list)
        return SameDimensionMapFunctor.same_dimension_map(x_shape, device, sample_dims, keep_ratio_list, sample_index_list, raw_sample_shape)
