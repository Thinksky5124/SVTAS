'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:27:09
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 14:37:18
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/deepspeed_model_pipline.py
'''
from typing import Dict
import torch
from .torch_model_pipline import TorchModelPipline
from svtas.utils import AbstractBuildFactory
import deepspeed
from deepspeed import comm as dist

@AbstractBuildFactory.register('model_pipline')
class DeepspeedModelPipline(TorchModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 device=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None,
                 ds_config={}) -> None:
        super().__init__(model, post_processing, device, criterion, optimizer,
                         lr_scheduler, pretrained)
        self.ds_config = ds_config
        self.ds_config['local_rank'] = self.local_rank
        deepspeed.init_distributed()
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        self.model, self.optimizer, _, self.lr_scheduler = deepspeed.initialize(args = ds_config,
                             model = self.model,
                             optimizer = self.optimizer,
                             model_parameters = self.model.parameters(),
                             lr_scheduler = self.lr_scheduler)
    
    def memory_clear(self):
        self.model.module._clear_memory_buffer()
    
    def resert_model_pipline(self, *args, **kwargs):
        dist.barrier()
        return super().resert_model_pipline(*args, **kwargs)
    
    @torch.no_grad()
    def output_post_processing(self, cur_vid, model_outputs=None, input_data=None):
        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        collect_dict = dict(
            predict=pred_cls_list,
            output_np=pred_score_list,
            ground_truth=ground_truth_list,
            vid=self.current_step_vid_list
        )
        gather_objects = [collect_dict for _ in range(self.nprocs)] # any picklable object
        output_list = [None for _ in range(self.nprocs)]
        dist.all_gather(output_list, gather_objects[dist.get_rank()])
        # collect
        pred_cls_list_i = []
        pred_score_list_i = []
        ground_truth_list_i = []
        vid_i = []
        for output_dict in output_list:
            pred_cls_list_i = pred_cls_list_i + output_dict["predict"]
            pred_score_list_i = pred_score_list_i + output_dict["output_np"]
            ground_truth_list_i = ground_truth_list_i + output_dict["ground_truth"]
            vid_i = vid_i + output_dict["vid"]
        outputs = dict(predict=pred_cls_list_i,
                        output_np=pred_score_list_i,
                        groundtruth=ground_truth_list)
        ground_truth_list = ground_truth_list_i
        vid = vid_i

        output_dict = dict(
            vid=vid,
            outputs=outputs,
            ground_truth=ground_truth_list
        )
        dist.barrier()
        return output_dict
    
    def to(self, device):
        pass

    def backward(self, loss_dict):
        loss = loss_dict["loss"]
        self.model.backward(loss)
    
    def clear_param_grad(self):
        pass

    def save(self) -> Dict:
        return {}

    def load(self, param_dict: Dict) -> None:
        pass
    
    def update_model_param(self):
        # actually_update function
        def actually_update():
            self.model.step()
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
    
    def end_model_pipline(self):
        dist.barrier()
        dist.destroy_process_group()
        return super().end_model_pipline()