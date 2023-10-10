'''
Author       : Thyssen Wen
Date         : 2023-10-06 15:16:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-09 10:23:18
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_model_ddp_pipline.py
'''
import os
from typing import Any, Dict
from svtas.utils.logger import AverageMeter
from svtas.utils import AbstractBuildFactory
from svtas.utils.misc import set_property
from svtas.optimizer.grad_clip import GradAccumulate, GradClip
from .torch_model_pipline import TorchModelPipline
from torch.nn.parallel import DistributedDataParallel

import torch
import torch.distributed as dist

@AbstractBuildFactory.register('model_pipline')
class TorchDistributedDataParallelModelPipline(TorchModelPipline):
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
        super().__init__(model, post_processing, device, criterion, optimizer,
                         lr_scheduler, pretrained, amp, grad_clip, grad_accumulate)
        # Todo 1.: Distributed Module Replace e.g.: batchnorm
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        dist.init_process_group("nccl", init_method="tcp://" + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT'], rank=self.rank, world_size=self.world_size)
        self.model = DistributedDataParallel(self.model, device_ids=[self.rank])

    def memory_clear(self):
        self.model.module._clear_memory_buffer()
    
    def resert_model_pipline(self, *args, **kwargs):
        torch.distributed.barrier()
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
        dist.all_gather_object(output_list, gather_objects[dist.get_rank()])
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
        torch.distributed.barrier()
        return output_dict
    
    def end_model_pipline(self):
        torch.distributed.barrier()
        dist.destroy_process_group()
        return super().end_model_pipline()
    