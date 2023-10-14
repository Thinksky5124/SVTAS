'''
Author       : Thyssen Wen
Date         : 2023-10-12 16:40:41
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 19:04:48
Description  : file content
FilePath     : /SVTAS/svtas/model/tas/tas_diffusion.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List
from ..architectures import DiffusionModel
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TemporalActionSegmentationDiffusionModel(DiffusionModel):
    def __init__(self,
                 vae: Dict,
                 scheduler: Dict,
                 unet: Dict = None,
                 weight_init_cfg: dict | List[dict] | None = dict(
                     vae=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))) -> None:
        super().__init__(vae, scheduler, unet, weight_init_cfg)
    
    def run_train(self, data_dict: Dict[str, torch.FloatTensor]):
        if self.vae is not None:
            latents_dict = self.vae.encode(data_dict)
        else:
            latents_dict = data_dict

        labels = torch.permute(data_dict['labels_onehot'], dims=[0, 2, 1]).contiguous()
        condition_latents = latents_dict['output_feature']

        # Sample noise to add to the images
        noise = torch.randn(labels.shape).to(labels.device)
        bs = labels.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.scheduler.num_train_timesteps, (bs,), device=labels.device
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_labels = self.scheduler.add_noise(labels, noise, timesteps)

        pred_noise_data_dict = dict(
            timestep = timesteps,
            noise_label = noisy_labels,
            condition_latens = condition_latents,
            labels = data_dict['labels'],
            boundary_prob = data_dict['boundary_prob']
        )
        noise_labels = self.unet(pred_noise_data_dict)['output']
        output_dict = dict(
            output = noise_labels,
            backbone_score = latents_dict['output'].squeeze(0)
        )
        return output_dict
    
    def run_test(self, data_dict: Dict[str, torch.FloatTensor]):
        # get latents
        if self.vae is not None:
            latents_dict = self.vae.encode(data_dict)
        else:
            latents_dict = data_dict
        
        # prepare for backward denoise
        
        condition_latents = latents_dict['output_feature']
        gt_labels = data_dict['labels_onehot']
        # Sample noise to add to the images
        pred_labels = torch.permute(torch.randn_like(gt_labels), dims=[0, 2, 1]).contiguous()
        self.scheduler.set_timesteps(device=pred_labels.device)
        generator = torch.Generator(pred_labels.device).manual_seed(self.scheduler.infer_region_seed)
        # denoise process
        for i, t in enumerate(self.scheduler.timesteps):
            t_now = torch.full((1,), t[0], device=t[0].device, dtype=torch.long)
            t_next = torch.full((1,), t[1], device=t[1].device, dtype=torch.long)

            pred_labels_scale = self.scheduler.scale_model_output(pred_labels)
            denoise_data_dict = dict(
                timestep = t_now,
                noise_label = pred_labels_scale,
                condition_latens = condition_latents
            )
            pred_labels_dict = self.unet(denoise_data_dict)
            pred_noise_labels = pred_labels_dict['output'].squeeze(0)
            
            step_dict = self.scheduler.step(model_output=pred_noise_labels,
                                            sample=pred_labels,
                                            timestep=t_now,
                                            next_timestep=t_next,
                                            generator=generator)
            pred_labels = step_dict['denoise_labels']
        pred_labels = self.scheduler.scale_model_output(pred_labels)
        decode_latents_dict = dict(
            output = pred_labels.unsqueeze(0),
            backbone_score = latents_dict['output'].squeeze(0)
        )
        # latent to output
        output_dict = self.vae.decode(decode_latents_dict)
        return output_dict