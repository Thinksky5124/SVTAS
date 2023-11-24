'''
Author       : Thyssen Wen
Date         : 2023-10-21 10:38:21
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 16:57:40
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch/torch_infer_model_pipline.py
'''
import torch
from typing import Dict
from svtas.model_pipline.torch_utils import load_state_dict

from ..base_pipline import BaseInferModelPipline
from svtas.utils import AbstractBuildFactory, get_logger
from svtas.utils.logger import get_root_logger_instance

@AbstractBuildFactory.register('model_pipline')
class TorchInferModelPipline(BaseInferModelPipline):
    """
    Args:
        compile_cfg: Dict, wheather to use torch.compile
    
    Example:
    ```
    compile_cfg = {
        "enable" = True
    }
    ```
    """
    def __init__(self,
                 model: Dict,
                 post_processing: Dict,
                 device=None,
                 compile_cfg: Dict = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        super().__init__(model, post_processing, device)
        self.compile_cfg = compile_cfg
        if compile_cfg is not None:
            if compile_cfg['enable']:
                logger = get_root_logger_instance()
                logger.info("Use torch.compile to accelerate......")
                copy_cfg = compile_cfg.copy()
                copy_cfg.pop('enable')
                self.model = torch.compile(self.model, **copy_cfg)
                logger.info("Finish torch.compile for model.")
    
    def load_from_ckpt_file(self, ckpt_path: str = None):
        ckpt_path = self.load_from_ckpt_file_ckeck(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        state_dicts = checkpoint["model_state_dict"]
        logger = get_logger("SVTAS")
        load_state_dict(self.model, state_dicts, logger=logger)

    def to(self, device):
        super().to(device=device)
        self.model.to(device)
    
    @torch.inference_mode()
    def forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                if torch.cuda.is_available():
                    input_data[key] = value.to(self.device)
                else:
                    input_data[key] = value
        outputs = self.model(input_data)
        return outputs, input_data
    
    def load(self, param_dict: Dict) -> None:
        self.model.load_state_dict(param_dict['model_state_dict'])
    
    def save(self) -> Dict:
        save_dict = {}
        save_dict['model_state_dict'] = self.model.state_dict()
        return save_dict