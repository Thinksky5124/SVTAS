'''
Author       : Thyssen Wen
Date         : 2023-10-11 23:10:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-23 23:18:49
Description  : file content
FilePath     : /SVTAS/svtas/model/scheduler/diffact_scheduler.py
'''
import math
from typing import Dict, Optional, Union, List
import torch
import torch.nn.functional as F
import numpy as np

from svtas.utils import AbstractBuildFactory
from .base_scheduler import BaseDiffusionScheduler

@AbstractBuildFactory.register('diffusion_scheduler')
class DiffsusionActionSegmentationScheduler(BaseDiffusionScheduler):
    """
    Diffusion Action Segmentation ref:https://arxiv.org/pdf/2303.17959.pdf,
    it is modify from DDIM scheduler
    """
    num_inference_steps: int
    def __init__(self,
                 num_train_timesteps: int = 1000,
                 num_inference_steps: int = 25,
                 ddim_sampling_eta: float = 1.0,
                 snr_scale: float = 0.5,
                 timestep_spacing: str = 'linspace',
                 infer_region_seed: int = 8) -> None:
        super().__init__(num_train_timesteps, num_inference_steps, infer_region_seed)
        self.ddim_sampling_eta = ddim_sampling_eta
        self.snr_scale = snr_scale
        self.timestep_spacing = timestep_spacing

        betas = self.cosine_beta_schedule(num_train_timesteps)  # torch.Size([1000])
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        self.timesteps = torch.from_numpy(np.arange(0, num_train_timesteps)[::-1].copy())
        
    @staticmethod
    def cosine_beta_schedule(timesteps, s=0.008):
        """
        cosine schedule
        as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
        """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)
    
    @staticmethod
    def extract(a, t, x_shape):
        """extract the appropriate  t  index for a batch of indices"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))
    
    def set_num_inference_steps(self, num_inference_steps: int = None):
        super().set_num_inference_steps(num_inference_steps)
        
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
            # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
            times = torch.flip(torch.linspace(0, self.num_train_timesteps - 1, steps=num_inference_steps + 1), dims=[0]).int().to(device)
            # [999, 749, 499, 249, -1]
            time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
            # [(999, 749), (749, 499), (499, 249), (249, -1)]

        self.timesteps = time_pairs
        
    def scale_model_input(self, sample: Dict[str, torch.FloatTensor], timestep: int | None = None) -> Dict[str, torch.FloatTensor]:
        sample = (sample * 2 - 1.) * self.snr_scale
        return torch.clamp(sample, min=-1 * self.snr_scale, max=self.snr_scale)

    def scale_model_output(self, sample: Dict[str, torch.FloatTensor], timestep: int | None = None) -> Dict[str, torch.FloatTensor]:
        sample = torch.clamp(sample, min=-1 * self.snr_scale, max=self.snr_scale) # [-scale, +scale]
        return ((sample / self.snr_scale) + 1) / 2

    def step(self, model_output: torch.FloatTensor, timestep: int, sample: torch.FloatTensor, next_timestep: int | None = None, generator=None) -> Dict:

        model_output = F.softmax(model_output, 1)
        assert(model_output.max() <= 1 and model_output.min() >= 0)

        model_output = self.scale_model_input(model_output, self.snr_scale)                              # [-scale, +scale]
        
        pred_noise = (
            (self.extract(self.sqrt_recip_alphas_cumprod, timestep, sample.shape) * sample - model_output) /
            self.extract(self.sqrt_recipm1_alphas_cumprod, timestep, sample.shape)
        )
        
        alpha = self.alphas_cumprod[timestep]
        alpha_next = self.alphas_cumprod[next_timestep]

        sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt() if (1 - alpha_next - sigma ** 2) >= 0 else 0

        noise = torch.randn(list(sample.shape), device=sample.device, dtype=sample.dtype, generator=generator)

        denoise_labels = model_output * alpha_next.sqrt() + \
                c * pred_noise + \
                sigma * noise
        output_dict = dict(
            denoise_labels = denoise_labels
        )
        return output_dict
    
    def add_noise(self, original_samples: torch.FloatTensor, noise: torch.FloatTensor, timesteps: torch.IntTensor) -> torch.FloatTensor:

        original_samples = self.scale_model_input(original_samples, timestep=timesteps)

        # noise sample
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, timesteps, original_samples.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, timesteps, original_samples.shape)

        noise_labels = sqrt_alphas_cumprod_t * original_samples + sqrt_one_minus_alphas_cumprod_t * noise

        noise_labels = torch.clamp(noise_labels, min=-1 * self.snr_scale, max=self.snr_scale)
        noise_labels = ((noise_labels / self.snr_scale) + 1) / 2.           # normalized [0, 1]

        return noise_labels