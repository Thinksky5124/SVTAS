'''
Author       : Thyssen Wen
Date         : 2022-11-23 20:14:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-01 15:10:39
Description  : Stochastic Backpropagation Decorator Module And Test Script
FilePath     : /SVTAS/svtas/utils/sbp.py
'''
import torch
import torch.nn as nn
from typing import Any
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

def generate_grad_mask(grad_mask, mode='uniform'):
    if mode == 'uniform':
        pass
    elif mode == 'random':
        pass
    
def grad_mask_sample(x: torch.Tensor, keep_ratio: int, return_mask: bool = True):
    if len(x.shape) == 2:
        BT, C = x.shape
        d_BT = BT // keep_ratio
        grad_mask = torch.zeros([BT], dtype=torch.bool).to(x.device)
        grad_mask[::keep_ratio] = True
        grad_mask = torch.roll(grad_mask, keep_ratio // 2, -1)  # roll to the center.
        grad_mask = grad_mask.view(BT, 1)
        x_w_grad = x.masked_select(grad_mask).view(d_BT, C)
    elif len(x.shape) == 3:
        B, C, T = x.shape
        d_T = T // keep_ratio
        grad_mask = torch.zeros([B, T], dtype=torch.bool).to(x.device)
        grad_mask[:, ::keep_ratio] = True
        grad_mask = torch.roll(grad_mask, keep_ratio // 2, -1)  # roll to the center.
        grad_mask = grad_mask.view(B, 1, T)
        x_w_grad = x.masked_select(grad_mask).view(B, C, d_T)
    elif len(x.shape) == 4:
        BT, C, H, W = x.shape
        d_BT = BT // keep_ratio
        grad_mask = torch.zeros([BT], dtype=torch.bool).to(x.device)
        grad_mask[::keep_ratio] = True
        grad_mask = torch.roll(grad_mask, keep_ratio // 2, -1)  # roll to the center.
        grad_mask = grad_mask.view(BT, 1, 1, 1)
        x_w_grad = x.masked_select(grad_mask).view(d_BT, C, H, W)
    elif len(x.shape) == 5:
        B, C, T, H, W = x.shape
        d_T = T // keep_ratio
        grad_mask = torch.zeros([B, T], dtype=torch.bool).to(x.device)
        grad_mask[:, ::keep_ratio] = True
        grad_mask = torch.roll(grad_mask, keep_ratio // 2, -1)  # roll to the center.
        grad_mask = grad_mask.view(B, 1, T, 1, 1)
        x_w_grad = x.masked_select(grad_mask).view(B, C, d_T, H, W)
    else:
        raise NotImplementedError
    if not return_mask:
        return x_w_grad
    return x_w_grad, grad_mask

class SBPOP(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, grad_mask_fn, forward_fn, args_len: int, *input: Any) -> torch.Tensor:
        args = input[:args_len]
        params = input[args_len:]
        
        with torch.no_grad():
            y = forward_fn(*args)
        
        input_w_grad, grad_mask_tuple, keep_ratio = grad_mask_fn(*args)
        ctx.save_for_backward(*input_w_grad)
        ctx.save_forward = forward_fn
        ctx.grad_mask_tuple = grad_mask_tuple
        ctx.params = params
        ctx.keep_ratio = keep_ratio
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
        for dy in grad_outputs:
            dy = grad_mask_sample(dy, keep_ratio=ctx.keep_ratio, return_mask=False)
            grad_outputs_list.append(dy)
        input_grads = list(torch.autograd.grad(y, (input_w_grad_list[0],) + ctx.params, grad_outputs_list))

        input_grad_list = []
        for dx_raw, dy, grad_mask in zip(input_grads, grad_outputs, ctx.grad_mask_tuple):
            dx = torch.zeros_like(dy)
            dx = dx.masked_scatter_(grad_mask, dx_raw)
            input_grad_list.append(dx)
        
        input_grad_list += [None]*(ctx.args_len - len(input_grad_list))

        for param_grad in input_grads[len(grad_outputs):]:
            input_grad_list.append(param_grad)

        del ctx.save_forward, ctx.keep_ratio, ctx.grad_mask_tuple, ctx.params, ctx.args_len
        return (None, None, None) + tuple(input_grad_list)

class StochasticBackpropagation(object):
    """Stochastic Backpropagation Class
    
    This class will overwrite nn.Module class to support stochastic backpropagation.
    It can use as a decorator or funtion.
    
    For example:

    case 1:
    ``` 
    @StochasticBackpropagation()
    class Module(nn.Module):
        ....
    ```
    case 2:
    ``` 
    sbp = StochasticBackpropagation()
    sbo_module = sbp(Module)
    ```
    """
    def __init__(self,
                 keep_ratio: float = 0.125) -> None:
        assert keep_ratio <= 1., f"keep_ratio must not grate than 1., now {keep_ratio:.2f}."
        self.keep_ratio = int(1 / keep_ratio)

    def __call__(decorate_self, Module: nn.Module) -> nn.Module:
        @wraps(Module)
        class SBPModule(Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)

            def grad_mask_fn(self, *input: Any):
                """generate mask for grad drop.
                """
                input_w_grad = []
                grad_mask_tuple = []
                for x in input:
                    x_w_grad, grad_mask = grad_mask_sample(x, decorate_self.keep_ratio, return_mask=True)
                    input_w_grad.append(x_w_grad)
                    grad_mask_tuple.append(grad_mask)
                input_w_grad = tuple(input_w_grad)
                grad_mask_tuple = tuple(grad_mask_tuple)

                return input_w_grad, grad_mask_tuple, decorate_self.keep_ratio

            def forward(self, *args: Any):
                args_len = len(args)
                return SBPOP.apply(self.grad_mask_fn, super().forward, args_len, *args, *tuple(self.parameters()))
        return SBPModule
