'''
Author       : Thyssen Wen
Date         : 2022-11-23 20:14:17
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-11 15:39:20
Description  : Stochastic Backpropagation Decorator Module
FilePath     : /SVTAS/svtas/utils/sbp/sbp.py
'''
import abc
import copy
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from ..logger import get_logger
from typing import Any, List, Dict, Tuple
from functools import partial
from .map_func import MaskMappingFunctor, SameDimensionMapFunctor

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
    """Stochastic BackPropagation Operator

    This class will be instantiated for module which will be apply for stochastic backpropagation.
    """
    @staticmethod
    def forward(ctx: Any, forward_fn: Any, grad_mask_generate_functor: MaskMappingFunctor, args_len: int, *input: Any) -> torch.Tensor:
        args = input[:args_len]
        params = input[args_len:]
        
        with torch.no_grad():
            y = forward_fn(*args)
        
        input_w_grad, grad_mask_tuple, sample_index_tuple = \
            StochasticBackPropagation.forward_grad_mask_fn(grad_mask_generate_functor.x_map_feature_fn, *args)
        ctx.save_for_backward(*input_w_grad)
        ctx.save_forward = forward_fn
        ctx.backward_grad_mask_generate_fn = grad_mask_generate_functor.feature_map_y_fn
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

        grad_outputs_list = StochasticBackPropagation.backward_grad_mask_fn(
                            ctx.backward_grad_mask_generate_fn,
                            ctx.sample_index_tuple,
                            ctx.grad_mask_tuple,
                            grad_outputs)
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
        del ctx.save_forward, ctx.grad_mask_tuple, ctx.sample_index_tuple, ctx.params, ctx.args_len, ctx.backward_grad_mask_generate_fn
        return (None, None, None) + tuple(input_grad_list)

class StochasticBackPropagation(object):
    """Stochastic Back Propagation Decorator
    
    This class will overwrite nn.Module class to support stochastic backpropagation.
    It can use as a decorator or funtion.
    
    You can refer paper from follows:
    
    [1]Stochastic Backpropagation: A Memory Efficient Strategy for Training Video Models, https://arxiv.org/pdf/2203.16755.pdf
    
    [2]An In-depth Study of Stochastic Backpropagation, https://arxiv.org/pdf/2210.00129.pdf

    Stochastic Back Propagation Class Flow Design Concept:
    ```
        x    -  model_forward    -> feature      -                                         sbp_module_forward                               -> y
        mask -  x_map_feature_fn -> feature_mask -                                         feature_map_y_fn                                 -> y_mask
        dx   <- backward          - d_feature <- feature_mask_select - d_feature_sample <- sbp_module_backward <- dy_sample <- y_mask_select - dy
        
    ```

    Args: 
        keep_ratio_list: List[float]
        sample_dims: List[int]
        grad_mask_mode: Literal
        register_sbp_module_dict: Dict[Any, Dict[str, Any]]
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
        sbp_module = sbp(Module)
    ```
    case 3:
    ```
        model = Model(**cfg)
        sbp = StochasticBackPropagation()
        sbp_module = sbp.register_module_from_instance(model, cfg)
    ```
    """
    SBP_ARGUMENTS = ('sbp_build', 'keep_ratio_list', 'grad_mask_mode_lsit', 'sample_dims', 'register_sbp_module_dict')
    KEEP_RATIO_LIST = [1]
    GRAD_MASK_MODE_LIST = 'uniform'
    SAMPLE_DIMS = [0]
    SUPPORT_SAMPLE_MODE = ['uniform', 'random', 'random_shift', 'uniform_random']
    REGISTER_SBP_MODULE_DICT = {}

    SAMPLE_INDEX_BUFFER = None
    CRITERION_SHAPE = None
    def __init__(self,
                 register_sbp_module_dict: Dict[Any, Dict[str, Any]] = {},
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
        StochasticBackPropagation.REGISTER_SBP_MODULE_DICT = register_sbp_module_dict
    
    @staticmethod
    def generate_sample(sample_mode, original_num, sample_num):
        if sample_mode == 'uniform':
            index = torch.arange(0, original_num, original_num // sample_num)
        elif sample_mode == 'random':
            sample_index = list(random.sample(list(range(0, original_num)), sample_num))
            sample_index.sort()
            index = torch.tensor(sample_index)
        elif sample_mode == 'random_shift':
            index = torch.arange(0, original_num, original_num // sample_num) + random.randint(a=0, b=original_num // sample_num)
        elif sample_mode == 'uniform_random':
            index = torch.arange(0, original_num, original_num // sample_num)
            index = index + torch.randint_like(index, high=original_num // sample_num)
        return index
    
    @staticmethod
    def grad_mask_sample(x: torch.Tensor,
                         generate_mask_fn: Any,
                         sample_index_list: List[List[int]],
                         raw_sample_shape: List[int],
                         return_mask: bool = False):
        grad_mask, match_sample_index_list, sample_shape = generate_mask_fn(x_shape=x.shape,
                                                                            device=x.device,
                                                                            sample_dims=StochasticBackPropagation.SAMPLE_DIMS,
                                                                            keep_ratio_list=StochasticBackPropagation.KEEP_RATIO_LIST,
                                                                            sample_index_list=sample_index_list,
                                                                            raw_sample_shape=raw_sample_shape)
        x = x.masked_select(grad_mask).reshape(sample_shape)
        if not return_mask:
            return x
        return x, grad_mask, match_sample_index_list

    @staticmethod
    def forward_grad_mask_fn(generate_mask_fn: Any, *input: Any):
        """generate mask for grad drop.
        """
        input_w_grad = []
        grad_mask_tuple = []
        sample_index_tuple = []
        for x in input:
            x_w_grad, grad_mask, sample_indexes = StochasticBackPropagation.grad_mask_sample(
                x=x,
                generate_mask_fn=generate_mask_fn,
                sample_index_list=StochasticBackPropagation.SAMPLE_INDEX_BUFFER,
                raw_sample_shape=StochasticBackPropagation.CRITERION_SHAPE,
                return_mask=True)
            input_w_grad.append(x_w_grad)
            grad_mask_tuple.append(grad_mask)
            sample_index_tuple.append(sample_indexes)
        input_w_grad = tuple(input_w_grad)
        grad_mask_tuple = tuple(grad_mask_tuple)
        sample_index_tuple = tuple(sample_index_tuple)

        return input_w_grad, grad_mask_tuple, sample_index_tuple
    
    @staticmethod
    def backward_grad_mask_fn(generate_mask_fn: Any, sample_index_tuple: Tuple[torch.Tensor], grad_mask_tuple: Tuple[torch.Tensor], grad_outputs: Any):
        grad_outputs_list = []
        for dy, sample_index_list, grad_mask in zip(grad_outputs, sample_index_tuple, grad_mask_tuple):
            dy = StochasticBackPropagation.grad_mask_sample(x=dy,
                                                            generate_mask_fn=generate_mask_fn,
                                                            sample_index_list=sample_index_list,
                                                            raw_sample_shape=list(grad_mask.shape),
                                                            return_mask=False)
            grad_outputs_list.append(dy)
        return grad_outputs_list

    @staticmethod
    def generate_x_grad_mask_index(module, input):
        x_keep_shape = list(input[0].shape)
        StochasticBackPropagation.CRITERION_SHAPE = x_keep_shape.copy()

        StochasticBackPropagation.SAMPLE_INDEX_BUFFER = []
        # reset grad mask index
        for dim, keep_ratio, sample_mode in zip(StochasticBackPropagation.SAMPLE_DIMS, StochasticBackPropagation.KEEP_RATIO_LIST,
                                   StochasticBackPropagation.GRAD_MASK_MODE_LIST):
            T = x_keep_shape[dim]
            d_T = 1 if T // keep_ratio <= 0 else T // keep_ratio
            x_keep_shape[dim] = d_T
            
            index = StochasticBackPropagation.generate_sample(sample_mode=sample_mode, original_num=T, sample_num=d_T)
            StochasticBackPropagation.SAMPLE_INDEX_BUFFER.append(index)

    def register_module_according_str(self, Model: nn.Module, register_dict: Dict[Any, MaskMappingFunctor]) -> nn.Module:
        logger = get_logger()
        for replace_key, grad_mask_generate_functor in register_dict.items():
            tokens = replace_key.split('.')
            sub_tokens = tokens[:-1]
            cur_mod = Model
            for s in sub_tokens:
                cur_mod = getattr(cur_mod, s)
            need_replace_module = getattr(cur_mod, tokens[-1])
            sbp_sub_module = self._proxy_instance(need_replace_module, grad_mask_generate_functor)
            setattr(cur_mod, tokens[-1], sbp_sub_module)
            logger.info(f"Module: {replace_key} are registed for stochastic back propagation train.")
        Model.register_forward_pre_hook(self.generate_x_grad_mask_index)
        return Model

    def register_module_according_nn(self, Model: nn.Module, register_dict: Dict[Any, Dict[str, Any]]) -> nn.Module:
        change_keys_dict = {}
        for name, module in Model.named_modules():
            for instance_key, map_fn_dict in register_dict.items():
                if isinstance(module, instance_key):
                    change_keys_dict[name] = map_fn_dict
        
        return self.register_module_according_str(Model=Model, register_dict=change_keys_dict)
    
    def _proxy_instance(self, Module: nn.Module, grad_mask_generate_functor: MaskMappingFunctor):
        class SBPProxyModule(nn.Module):
            def __init__(self, module) -> None:
                super().__init__()
                # proxy all paramters
                self._modules = module._modules
                self._buffers = module._buffers
                self._parameters = module._parameters
                # aggregate pricipal class
                self.__dict__.update({"_sbp_principal_module":module})
                        
            def forward(self, *args: Any):
                args_len = len(args)
                return StochasticBackPropagationOperator.apply(self._sbp_principal_module.forward,
                                    grad_mask_generate_functor, args_len, *args, *tuple(self.parameters()))
        proxy_module = SBPProxyModule(Module)
        return proxy_module

    def register_module_from_instance(self, Model: nn.Module, cfg: dict):
        logger = get_logger()
        logger.info("Use Stochastic Back Propagation For Model Train.")
        keys_list = list(StochasticBackPropagation.REGISTER_SBP_MODULE_DICT.keys())
        if len(keys_list) > 0:
            if isinstance(keys_list[0], str):
                return self.register_module_according_str(Model=Model(**cfg), register_dict=StochasticBackPropagation.REGISTER_SBP_MODULE_DICT)
            elif issubclass(keys_list[0], nn.Module):
                return self.register_module_according_nn(Model=Model(**cfg), register_dict=StochasticBackPropagation.REGISTER_SBP_MODULE_DICT)
            else:
                raise NotImplementedError("Not support sbp registor module!")
        else:
            sbp_module = self(Model)
            return sbp_module(**cfg)

    def __call__(decorate_self, Module: nn.Module) -> nn.Module:
        same_dimension_map_functor = SameDimensionMapFunctor()
        @wraps(Module)
        class SBPModule(Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                self.register_forward_pre_hook(decorate_self.generate_x_grad_mask_index)

            def forward(self, *args: Any):
                args_len = len(args)
                return StochasticBackPropagationOperator.apply(super().forward,
                                same_dimension_map_functor,
                                args_len, *args, *tuple(self.parameters()))
        return SBPModule