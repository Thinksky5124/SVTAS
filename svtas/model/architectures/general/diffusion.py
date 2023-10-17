'''
Author       : Thyssen Wen
Date         : 2023-09-21 15:56:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-17 10:16:22
Description  : file content
FilePath     : /SVTAS/svtas/model/architectures/general/diffusion.py
'''
import torch
import inspect
from typing import Any, List, Dict, Optional, Union
from .vae import VariationalAutoEncoder
from ...scheduler import BaseDiffusionScheduler
from ...unet import ConditionUnet
from svtas.utils import AbstractBuildFactory
from svtas.model_pipline import TorchBaseModel

@AbstractBuildFactory.register('model')
class DiffusionModel(TorchBaseModel):
    vae: VariationalAutoEncoder
    scheduler: BaseDiffusionScheduler
    unet: ConditionUnet

    def __init__(self,
                 vae: Dict,
                 scheduler: Dict,
                 unet: Dict = None,
                 weight_init_cfg: dict | List[dict] | None = dict(
                     vae=dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]
                    ))) -> None:
        super().__init__(weight_init_cfg)
        self.component_list = []
        if vae is not None:
            self.vae = AbstractBuildFactory.create_factory('model').create(vae)
            self.component_list.append('vae')
        else:
            self.vae = None
        
        if scheduler is not None:
            self.scheduler = AbstractBuildFactory.create_factory('diffusion_scheduler').create(scheduler)
        else:
            self.scheduler = None
        
        if unet is not None:
            self.unet = AbstractBuildFactory.create_factory('model').create(unet)
            self.component_list.append('unet')
        else:
            self.unet = None
    
    def init_weights(self, init_cfg: dict = {}):
        if len(init_cfg) <= 0 or init_cfg is None:
            init_cfg = self.weight_init_cfg
        for component_name in self.component_list:
            if component_name in init_cfg.keys():
                getattr(self, component_name).init_weights(init_cfg[component_name])
            else:
                getattr(self, component_name).init_weights()
    
    def _clear_memory_buffer(self):
        for component_name in self.component_list:
            getattr(self, component_name)._clear_memory_buffer()
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self,
                                  generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
                                  eta: float = 0.0):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
