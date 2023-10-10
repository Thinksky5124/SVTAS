'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:24:52
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 14:04:23
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_model_pipline.py
'''
from typing import Any, Dict
from svtas.utils.logger import AverageMeter
from .base_pipline import BaseModelPipline
from svtas.utils import AbstractBuildFactory, get_logger
from svtas.utils.misc import set_property
from svtas.optimizer.grad_clip import GradAccumulate, GradClip

import torch
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp.grad_scaler import GradScaler
from svtas.model_pipline.torch_utils import load_state_dict

@AbstractBuildFactory.register('model_pipline')
class TorchModelPipline(BaseModelPipline):
    scaler: GradScaler
    grad_accumulate: GradAccumulate
    grad_clip: GradClip

    def __init__(self,
                 model,
                 post_processing,
                 device=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None,
                 amp: Dict = None,
                 grad_clip: Dict = None,
                 grad_accumulate: Dict = None) -> None:
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
        super().__init__(model, post_processing, device, criterion, optimizer, lr_scheduler, pretrained)
        set_property(self, 'grad_clip', grad_clip, 'optimizer')
        set_property(self, 'grad_accumulate', grad_accumulate, 'optimizer')

        self.use_amp = False
        if amp is not None:
            self.scaler = GradScaler(**amp)
            self.use_amp = True
    
    def load_from_ckpt_file(self, ckpt_path: str = None):
        ckpt_path = self.load_from_ckpt_file_ckeck(ckpt_path)
        checkpoint = torch.load(ckpt_path)
        state_dicts = checkpoint["model_state_dict"]
        logger = get_logger("SVTAS").logger
        load_state_dict(self.model, state_dicts, logger=logger)

    def to(self, device):
        super().to(device=device)
        if self.optimizer is not None:
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
    
    def forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                if torch.cuda.is_available():
                    input_data[key] = value.to(self.device)
                else:
                    input_data[key] = value
        if not self.grad_accumulate:
            input_data['precise_sliding_num'] = torch.ones_like(input_data['precise_sliding_num'])

        outputs = self.model(input_data)
        return outputs, input_data
    
    @torch.no_grad()
    def init_post_processing(self, input_data) -> None:
        vid_list = input_data['vid_list']
        sliding_num = input_data['sliding_num']
        if len(vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

    @torch.no_grad()
    def update_post_processing(self, model_outputs, input_data) -> None:
        idx = input_data['current_sliding_cnt']
        labels = input_data['labels']
        score = model_outputs['output']
        with torch.no_grad():
            output = self.post_processing.update(score, labels, idx)
        return output

    @torch.no_grad()
    def output_post_processing(self, cur_vid, model_outputs = None, input_data = None):
        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                       output_np=pred_score_list)
        vid = cur_vid
        output_dict = dict(
            vid=vid,
            outputs=outputs,
            ground_truth=ground_truth_list
        )
        return output_dict
    
    @torch.no_grad()
    def direct_output_post_processing(self, cur_vid, model_outputs = None, input_data = None):
        # get pred result
        output_dict = self.post_processing.output()
        return output_dict
    
    def caculate_loss(self, outputs, input_data) -> dict:
        loss_dict = self.criterion(outputs, input_data)
        return loss_dict
    
    def backward(self, loss_dict):
        loss = loss_dict["loss"]
        if not self.use_amp:
            loss.backward()
        else:
            self.scaler.scale(loss).backward()

        if self.grad_clip is not None:
            for param_group in self.optimizer.param_groups:
                self.grad_clip(param_group['params'])
    
    def memory_clear(self):
        self.model._clear_memory_buffer()

    def clear_param_grad(self):
        self.optimizer.zero_grad()

    def update_model_param(self):
        # actually_update function
        def actually_update():
            if self.use_amp:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.clear_param_grad()
            self.memory_clear()

        # judge weather use grad_accumulate
        if self.grad_accumulate is not None:
            if self.grad_accumulate.update:
                actually_update()
            else:
                return
        else:
            actually_update()
    
    def init_model_param(self, init_cfg: Dict = {}):
        self.model.init_weights(init_cfg)
    
    def update_optim_policy(self):
        self.lr_scheduler.step()

    def resert_model_pipline(self, *args, **kwargs):
        return super().resert_model_pipline(*args, **kwargs)
    
    def end_model_pipline(self):
        return super().end_model_pipline()
    
    def save(self) -> Dict:
        save_dict = {}
        save_dict['model_state_dict'] = self.model.state_dict()
        if self.optimizer is not None:
            save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
        return save_dict

    def load(self, param_dict: Dict) -> None:
        self.model.load_state_dict(param_dict['model_state_dict'])
        self.optimizer.load_state_dict(param_dict['optimizer_state_dict'])
    
    def train_run(self, data_dict, is_end_step: bool = True):
        if self.use_amp:
            with autocast():
                outputs, input_data = self.forward(data_dict)
                loss_dict = self.caculate_loss(outputs=outputs, input_data=input_data)
        else:
            outputs, input_data = self.forward(data_dict)
            loss_dict = self.caculate_loss(outputs=outputs, input_data=input_data)

        self.backward(loss_dict=loss_dict)
        self.update_model_param()
        return outputs, loss_dict

    @torch.no_grad()
    def test_run(self, data_dict, is_end_step: bool = True):
        if self.use_amp:
            with autocast():
                outputs, input_data = self.forward(data_dict)
                if self.criterion is not None:
                    loss_dict = self.caculate_loss(outputs=outputs, input_data=input_data)
                else:
                    loss_dict = {}
        else:
            outputs, input_data = self.forward(data_dict)
            if self.criterion is not None:
                loss_dict = self.caculate_loss(outputs=outputs, input_data=input_data)
            else:
                loss_dict = {}

        return outputs, loss_dict
