'''
Author       : Thyssen Wen
Date         : 2022-11-23 20:14:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-07 22:17:43
Description  : Stochastic Backpropagation Decorator Module
FilePath     : /SVTAS/svtas/utils/sbp.py
'''
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List
from functools import partial

WRAPPER_ASSIGNMENTS = ('__module__', '__name__', '__qualname__', '__doc__',
                       '__annotations__')
WRAPPER_UPDATES = ('__dict__',)

def update_wrapper(wrapper,
                   wrapped,
                   assigned = WRAPPER_ASSIGNMENTS,
                   updated = WRAPPER_UPDATES):
    """Update a wrapper function to look like the wrapped function

       wrapper is the function to be updated
       wrapped is the original function
       assigned is a tuple naming the attributes assigned directly
       from the wrapped function to the wrapper function (defaults to
       functools.WRAPPER_ASSIGNMENTS)
       updated is a tuple naming the attributes of the wrapper that
       are updated with the corresponding attribute from the wrapped
       function (defaults to functools.WRAPPER_UPDATES)
    """
    for attr in assigned:
        try:
            value = getattr(wrapped, attr)
        except AttributeError:
            pass
        else:
            setattr(wrapper, attr, value)

    # Issue #17482: set __wrapped__ last so we don't inadvertently copy it
    # from the wrapped function when updating __dict__
    wrapper.__wrapped__ = wrapped
    # Return the wrapper so this can be used as a decorator via partial()
    return wrapper

def wraps(wrapped,
          assigned = WRAPPER_ASSIGNMENTS,
          updated = WRAPPER_UPDATES):
    """Decorator factory to apply update_wrapper() to a wrapper function

       Returns a decorator that invokes update_wrapper() with the decorated
       function as the wrapper argument and the arguments to wraps() as the
       remaining arguments. Default arguments are as for update_wrapper().
       This is a convenience function to simplify applying partial() to
       update_wrapper().
    """
    return partial(update_wrapper, wrapped=wrapped,
                   assigned=assigned, updated=updated)

class StochasticBackPropagationOperator(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, forward_fn, args_len: int, *input: Any) -> torch.Tensor:
        args = input[:args_len]
        params = input[args_len:]
        
        with torch.no_grad():
            y = forward_fn(*args)
        
        input_w_grad, grad_mask_tuple, sample_index_tuple = StochasticBackPropagation.grad_mask_fn(*args)
        ctx.save_for_backward(*input_w_grad)
        ctx.save_forward = forward_fn
        ctx.grad_mask_tuple = grad_mask_tuple
        ctx.sample_index_tuple = sample_index_tuple
        ctx.params = params
        ctx.args_len = args_len
        return y
    
    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> tuple:
        input_w_grad = ctx.saved_tensors

        input_w_grad_list = []
        for x in input_w_grad:
            x = x.detach().requires_grad_(True)
            input_w_grad_list.append(x)
        with torch.enable_grad():
            y = ctx.save_forward(*input_w_grad_list)

        grad_outputs_list = []
        for dy, sample_index_list, grad_mask in zip(grad_outputs, ctx.sample_index_tuple, ctx.grad_mask_tuple):
            assert len(list(y.shape)) == len(list(grad_mask.shape)), "x and y must be equal dim length"
            dy = StochasticBackPropagation.mismatch_grad_mask_sample(dy, sample_shape=list(y.shape), sample_index_list=sample_index_list, raw_sample_shape=list(grad_mask.shape))
            grad_outputs_list.append(dy)
        input_tuple = (input_w_grad_list[0], ) if len(input_w_grad_list) == 1 else tuple(input_w_grad_list[:ctx.args_len])
        input_grads = list(torch.autograd.grad(y, input_tuple + ctx.params, grad_outputs_list, allow_unused=True))

        input_grad_list = []
        for dx_raw, grad_mask in zip(input_grads, ctx.grad_mask_tuple):
            if dx_raw is not None:
                dx = torch.zeros(grad_mask.shape, dtype=dx_raw.dtype, device=dx_raw.device)
                dx = dx.masked_scatter_(grad_mask, dx_raw)
            input_grad_list.append(dx)

        for param_grad in input_grads[ctx.args_len:]:
            input_grad_list.append(param_grad)

        del ctx.save_forward, ctx.grad_mask_tuple, ctx.sample_index_tuple, ctx.params, ctx.args_len
        return (None, None) + tuple(input_grad_list)

class StochasticBackPropagation(object):
    """Stochastic Back Propagation Decorator
    
    This class will overwrite nn.Module class to support stochastic backpropagation.
    It can use as a decorator or funtion.
    
    You can refer paper from follows:
    
    [1]Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models, https://arxiv.org/pdf/2203.16755.pdf
    
    [2]An In-depth Study of Stochastic Backpropagation, https://arxiv.org/pdf/2210.00129.pdf

    Args: 
        keep_ratio_list: List[float]
        sample_dims: List[int]
        grad_mask_mode: Literal

    Usage:

    case 1:
    ``` 
    @StochasticBackPropagation()
    class Module(nn.Module):
        ...
    ```
    case 2:
    ``` 
    sbp = StochasticBackPropagation()
    sbo_module = sbp(Module)
    ```
    """
    SBP_ARGUMENTS = ('sbp_build', 'keep_ratio_list', 'grad_mask_mode_lsit', 'sample_dims')
    KEEP_RATIO_LIST = [1]
    GRAD_MASK_MODE_LIST = 'uniform'
    SAMPLE_DIMS = [0]
    SUPPORT_SAMPLE_MODE = ['uniform', 'random', 'random_shift', 'uniform_random', 'uniform_grid',
                            'uniform_grid_shift', 'random_grid']
    def __init__(self,
                 keep_ratio_list: List[float] = [0.125],
                 sample_dims: List[int] = [0],
                 grad_mask_mode_lsit: List[str] = ['uniform'],
                 **kwargs) -> None:
        if isinstance(keep_ratio_list, float):
            assert keep_ratio_list <= 1., f"keep_ratio must not grater than 1., now {keep_ratio_list:.2f}."
            keep_ratio_list = [keep_ratio_list]
        else:
            for keep_ratio in keep_ratio_list:
                assert keep_ratio <= 1., f"keep_ratio must not grater than 1., now {keep_ratio:.2f}."
        if len(keep_ratio_list) != len(sample_dims):
            if len(keep_ratio_list) == 1:
                keep_ratio_list = keep_ratio_list*len(sample_dims)
            else:
                raise ValueError("keep_ratio_list len must be equal to sample_dims len!")
        keep_ratio_list_t = []
        for keep_ratio in keep_ratio_list:
            keep_ratio_list_t.append(int(1 / keep_ratio))
        StochasticBackPropagation.KEEP_RATIO_LIST = keep_ratio_list_t

        if isinstance(grad_mask_mode_lsit, str):
            assert grad_mask_mode_lsit in StochasticBackPropagation.SUPPORT_SAMPLE_MODE, f"grad_mask_mode: {grad_mask_mode_lsit} is not support!"
            grad_mask_mode_lsit = [grad_mask_mode_lsit]
        else:
            for grad_mask_mode in grad_mask_mode_lsit:
                assert grad_mask_mode in StochasticBackPropagation.SUPPORT_SAMPLE_MODE, f"grad_mask_mode: {grad_mask_mode} is not support!"

        if len(grad_mask_mode_lsit) != len(sample_dims):
            if len(grad_mask_mode_lsit) == 1:
                grad_mask_mode_lsit = grad_mask_mode_lsit*len(sample_dims)
            else:
                raise ValueError("grad_mask_mode_lsit len must be equal to sample_dims len!")
        
        StochasticBackPropagation.GRAD_MASK_MODE_LIST = grad_mask_mode_lsit
        # sort dims accelerate sample
        sample_dims.sort()
        StochasticBackPropagation.SAMPLE_DIMS = sample_dims
    
    @staticmethod
    def generate_sample(sample_mode, original_num, sample_num):
        if sample_mode == 'uniform':
            index = torch.arange(0, original_num, original_num // sample_num)
        elif sample_mode == 'random':
            sample_index = list(random.sample(list(range(0, original_num)), sample_num))
            sample_index.sort()
            index = torch.tensor(sample_index)
        elif sample_mode == 'random_shift':
            index = torch.arange(0, original_num, original_num // sample_num) + torch.randint(0, high=original_num // sample_num)
        elif sample_mode == 'uniform_random':
            index = torch.arange(0, original_num, original_num // sample_num)
            index = index + torch.randint_like(index, high=original_num // sample_num)
        elif sample_mode == 'uniform_grid':
            #!
            index = torch.arange(0, original_num, original_num // sample_num)
            index = index + torch.randint_like(index, high=original_num // sample_num)
        elif sample_mode == 'uniform_grid_shift':
            #!
            index = torch.arange(0, original_num, original_num // sample_num)
            index = index + torch.randint_like(index, high=original_num // sample_num)
        elif sample_mode == 'random_grid':
            #!
            index = torch.arange(0, original_num, original_num // sample_num)
            index = index + torch.randint_like(index, high=original_num // sample_num)
        return index
    
    @staticmethod
    def mismatch_grad_mask_sample(x: torch.Tensor, sample_shape: List[int], sample_index_list: List[torch.Tensor], raw_sample_shape: List[int]):
        # regenrate
        grad_mask_false = torch.zeros_like(x, dtype=torch.bool)
        grad_mask_true = torch.ones_like(grad_mask_false)
        grad_mask = None

        for dim, sample_index in zip(StochasticBackPropagation.SAMPLE_DIMS, sample_index_list):

            index = torch.floor(sample_index * (x.shape[dim] / raw_sample_shape[dim]))
            sample_sample_index = torch.arange(0, index.shape[0], index.shape[0] // sample_shape[dim])
            index = index[sample_sample_index].long()

            index_dims = torch.ones(len(x.shape), dtype=torch.int32)
            index_dims[dim] = sample_shape[dim]
            index = index.reshape(index_dims.tolist())
            expand_dims = copy.deepcopy(list(x.shape))
            expand_dims[dim] = 1
            grad_mask_index = index.repeat(expand_dims).to(device=x.device)
            
            if grad_mask is None:
                grad_mask = torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)
            else:
                grad_mask = grad_mask * torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)

        grad_mask = grad_mask.bool()
        x = x.masked_select(grad_mask).reshape(sample_shape)
        return x
    
    @staticmethod
    def grad_mask_sample(x: torch.Tensor, return_mask: bool = True):

        grad_mask_false = torch.zeros_like(x, dtype=torch.bool)
        grad_mask_true = torch.ones_like(grad_mask_false)
        x_keep_shape = list(x.shape)
        grad_mask = None

        sample_index_list = []

        for dim, keep_ratio, sample_mode in zip(StochasticBackPropagation.SAMPLE_DIMS, StochasticBackPropagation.KEEP_RATIO_LIST,
                                   StochasticBackPropagation.GRAD_MASK_MODE_LIST):
            T = x_keep_shape[dim]
            d_T = 1 if T // keep_ratio <= 0 else T // keep_ratio
            x_keep_shape[dim] = d_T
            
            index = StochasticBackPropagation.generate_sample(sample_mode=sample_mode, original_num=T, sample_num=d_T)
            sample_index_list.append(index)

            index_dims = torch.ones(len(x.shape), dtype=torch.int32)
            index_dims[dim] = d_T
            index = index.reshape(index_dims.tolist())
            expand_dims = copy.deepcopy(list(x.shape))
            expand_dims[dim] = 1
            grad_mask_index = index.repeat(expand_dims).to(device=x.device)
            
            if grad_mask is None:
                grad_mask = torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)
            else:
                grad_mask = grad_mask * torch.scatter(grad_mask_false, dim, grad_mask_index, grad_mask_true)

        x_w_grad = x.masked_select(grad_mask).reshape(x_keep_shape)

        if not return_mask:
            return x_w_grad
        return x_w_grad, grad_mask, sample_index_list
    
    @staticmethod
    def grad_mask_fn(*input: Any):
        """generate mask for grad drop.
        """
        input_w_grad = []
        grad_mask_tuple = []
        sample_index_tuple = []
        for x in input:
            x_w_grad, grad_mask, sample_indexes = StochasticBackPropagation.grad_mask_sample(x, return_mask=True)
            input_w_grad.append(x_w_grad)
            grad_mask_tuple.append(grad_mask)
            sample_index_tuple.append(sample_indexes)
        input_w_grad = tuple(input_w_grad)
        grad_mask_tuple = tuple(grad_mask_tuple)
        sample_index_tuple = tuple(sample_index_tuple)

        return input_w_grad, grad_mask_tuple, sample_index_tuple

    def __call__(decorate_self, Module: nn.Module) -> nn.Module:
        @wraps(Module)
        class SBPModule(Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            def forward(self, *args: Any):
                args_len = len(args)
                return StochasticBackPropagationOperator.apply(super().forward, args_len, *args, *tuple(self.parameters()))
        return SBPModule
