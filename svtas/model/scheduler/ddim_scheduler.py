'''
Author       : Thyssen Wen
Date         : 2023-10-25 09:44:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-25 10:50:47
Description  : ref: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py
FilePath     : /SVTAS/svtas/model/scheduler/ddim_scheduler.py
'''
import math
from typing import Dict, Optional, Union, List
import torch
import torch.nn.functional as F
import numpy as np

from svtas.utils import AbstractBuildFactory
from .ddpm_scheduler import DDPMScheduler

@AbstractBuildFactory.register('diffusion_scheduler')
class DDIMScheduler(DDPMScheduler):
    """
    `DDIMScheduler` extends the denoising procedure introduced in denoising diffusion probabilistic models (DDPMs) with
    non-Markovian guidance.

    Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.
    """

    def __init__(self,
                 num_train_timesteps: int,
                 num_inference_steps: int,
                 infer_region_seed: int,
                 sampling_eta: float = 1.0,
                 timestep_spacing: str = 'linspace',
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 simple_var: bool = True) -> None:
        super().__init__(num_train_timesteps, num_inference_steps, infer_region_seed,
                         timestep_spacing, min_beta, max_beta, simple_var)
        self.sampling_eta = sampling_eta
        if self.simple_var:
            self.sampling_eta = 1.0
    
    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, next_timestep: int | None = None, generator=None) -> Dict:
        ab_cur = self.alpha_bars[timestep]
        ab_prev = self.alpha_bars[next_timestep] if next_timestep >= 0 else 1

        var = self.sampling_eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
        noise = torch.randn(list(sample.shape), device=sample.device, dtype=sample.dtype, generator=generator)

        first_term = (ab_prev / ab_cur)**0.5 * sample
        second_term = ((1 - ab_prev - var)**0.5 -
                        (ab_prev * (1 - ab_cur) / ab_cur)**0.5) * model_output
        if self.simple_var:
            third_term = (1 - ab_cur / ab_prev)**0.5 * noise
        else:
            third_term = var**0.5 * noise
        denoise_labels = first_term + second_term + third_term

        output_dict = dict(
            denoise_labels = denoise_labels
        )
        return output_dict