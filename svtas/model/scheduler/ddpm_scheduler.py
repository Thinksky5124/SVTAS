'''
Author       : Thyssen Wen
Date         : 2023-10-25 09:44:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-25 10:47:28
Description  : ref: https://github.com/SingleZombie/DL-Demos/blob/master/dldemos/ddim/ddpm.py
FilePath     : /SVTAS/svtas/model/scheduler/ddpm_scheduler.py
'''
import math
from typing import Dict, Optional, Union, List
import torch
import torch.nn.functional as F
import numpy as np

from svtas.utils import AbstractBuildFactory
from .base_scheduler import BaseDiffusionScheduler

@AbstractBuildFactory.register('diffusion_scheduler')
class DDPMScheduler(BaseDiffusionScheduler):
    def __init__(self,
                 num_train_timesteps: int,
                 num_inference_steps: int,
                 infer_region_seed: int,
                 timestep_spacing: str = 'linspace',
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02,
                 simple_var: bool = True) -> None:
        super().__init__(num_train_timesteps, num_inference_steps, infer_region_seed)
        self.timestep_spacing = timestep_spacing
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.simple_var = simple_var
        betas = torch.linspace(self.min_beta, self.max_beta, num_train_timesteps)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())

    def set_num_inference_steps(self, num_inference_steps: int = None):
        return super().set_num_inference_steps(num_inference_steps)

    def scale_model_input(self, sample: Dict[str, torch.FloatTensor], timestep: int | None = None) -> Dict[str, torch.FloatTensor]:
        return sample
    
    def scale_model_output(self, sample: Dict[str, torch.FloatTensor], timestep: int | None = None) -> Dict[str, torch.FloatTensor]:
        return sample
    
    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
        timesteps: Optional[List[int]] = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).

        Args:
            num_inference_steps (`int`):
                The number of diffusion steps used when generating samples with a pre-trained model. If used,
                `timesteps` must be `None`.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
            timesteps (`List[int]`, *optional*):
                Custom timesteps used to support arbitrary spacing between timesteps. If `None`, then the default
                timestep spacing strategy of equal spacing between timesteps is used. If `timesteps` is passed,
                `num_inference_steps` must be `None`.

        """
        if num_inference_steps is not None and timesteps is not None:
            raise ValueError("Can only pass one of `num_inference_steps` or `custom_timesteps`.")
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        
        if self.timestep_spacing == 'linspace':
            times = torch.flip(torch.linspace(0, self.num_train_timesteps - 1, steps=num_inference_steps + 1), dims=[0]).int().to(device)
            time_pairs = list(zip(times[:-1], times[1:]))

        self.timesteps = time_pairs

    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:
        alpha_bar = self.alpha_bars[timesteps].reshape(-1, 1, 1)
        noise_labels = noise * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * original_samples
        return noise_labels

    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, next_timestep: int | None = None, generator=None) -> Dict:        
        if timestep == 0:
            noise = 0
        else:
            if self.simple_var:
                var = self.betas[timestep]
            else:
                var = (1 - self.alpha_bars[timestep - 1]) / (
                    1 - self.alpha_bars[timestep]) * self.betas[timestep]
                
            noise = torch.randn(list(sample.shape), device=sample.device, dtype=sample.dtype, generator=generator)
            noise *= torch.sqrt(var)

        mean = (sample -
                (1 - self.alphas[timestep]) / torch.sqrt(1 - self.alpha_bars[timestep]) *
                model_output) / torch.sqrt(self.alphas[timestep])
        denoise_labels = mean + noise

        output_dict = dict(
            denoise_labels = denoise_labels
        )
        return output_dict