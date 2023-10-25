'''
Author       : Thyssen Wen
Date         : 2023-10-11 23:08:48
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-25 09:45:10
Description  : file content
FilePath     : /SVTAS/svtas/model/scheduler/base_scheduler.py
'''
import abc
import torch
from typing import Optional, Dict

class BaseDiffusionScheduler(torch.nn.Module, metaclass=abc.ABCMeta):
    """
    Base Class for Scheduler of DiffusionModel Model
    """
    timesteps: torch.LongTensor
    num_inference_steps: int
    SEED_RANGE = [-9999999999, 99999999999]
    def __init__(self,
                 num_train_timesteps: int,
                 num_inference_steps: int,
                 infer_region_seed: int) -> None:
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.infer_region_seed = infer_region_seed
        self.seed_generator = torch.Generator().manual_seed(self.infer_region_seed)
    
    def reset_state(self):
        if not self.training:
            self.seed_generator = torch.Generator().manual_seed(self.infer_region_seed)
    
    def get_random_seed_from_generator(self) -> int:
        return int(torch.randint(low=self.SEED_RANGE[0], high=self.SEED_RANGE[1], size=[1], generator=self.seed_generator).item())

    def scale_model_input(self, sample: Dict[str, torch.FloatTensor], timestep: Optional[int] = None) -> Dict[str, torch.FloatTensor]:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.FloatTensor`): input sample
            timestep (`int`, optional): current timestep

        Returns:
            `torch.FloatTensor`: scaled input sample
        """
        return sample
    
    def set_num_inference_steps(self, num_inference_steps: int = None):
        """
        Sets the discrete timesteps used for the diffusion chain. Supporting function to be run before inference.

        Note that this scheduler uses a slightly different step ratio than the other diffusers schedulers. The
        different step ratio is to mimic the original karlo implementation and does not affect the quality or accuracy
        of the results.

        Args:
            num_inference_steps (`int`):
                the number of diffusion steps used when generating samples with a pre-trained model.
        """
        if not num_inference_steps:
            self.num_inference_steps = num_inference_steps

    @abc.abstractmethod
    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: int,
        sample: torch.FloatTensor,
        prev_timestep: Optional[int] = None,
        generator=None
    ) -> Dict:
        """
        Predict the sample at the previous timestep by reversing the SDE. Core function to propagate the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`): direct output from learned diffusion model.
            timestep (`int`): current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                current instance of sample being created by diffusion process.
            prev_timestep (`int`, *optional*): The previous timestep to predict the previous sample at.
                Used to dynamically compute beta. If not given, `t-1` is used and the pre-computed beta is used.
            generator: random number generator.
            return_dict (`bool`): option for returning tuple rather than UnCLIPSchedulerOutput class

        Returns:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] or `tuple`:
            [`~schedulers.scheduling_utils.UnCLIPSchedulerOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is the sample tensor.

        """
        pass

    @abc.abstractmethod
    def add_noise(
        self,
        original_samples: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.IntTensor,
    ) -> torch.FloatTensor:
        pass