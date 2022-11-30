'''
Author       : Thyssen Wen
Date         : 2022-11-30 11:01:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-30 14:26:45
Description  : Check Close Function
FilePath     : /SVTAS/tests/common/check.py
'''
import torch
import numpy as np

def check_tensor_close(i_tensor: torch.Tensor, c_tensor: np.array, eps: float = 1e-5) -> bool:
    """Check Tensor Whether Close or not

    Args: 
        i_tensor: torch.Tensor
        c_tensor: np.array
        eps: float, default: 1e-5
    
    return:
        bool: True -> close, False -> not close
    """
    if torch.is_tensor(i_tensor):
        i_numpy_tensor = i_tensor.detach().cpu().data.numpy()
    else:
        i_numpy_tensor = i_tensor
    
    if torch.is_tensor(c_tensor):
        c_numpy_tensor = c_tensor.detach().cpu().data.numpy()
    else:
        c_numpy_tensor = c_tensor
    return np.allclose(i_numpy_tensor, c_numpy_tensor, rtol=eps)