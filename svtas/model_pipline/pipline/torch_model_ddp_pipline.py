'''
Author       : Thyssen Wen
Date         : 2023-10-06 15:16:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-15 14:43:00
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_model_ddp_pipline.py
'''
import os
from typing import Any, Dict
from svtas.utils import AbstractBuildFactory
from .torch_model_pipline import TorchModelPipline
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.optim import ZeroRedundancyOptimizer

import torch
import torch.distributed as dist

@AbstractBuildFactory.register('model_pipline')
class TorchDistributedDataParallelModelPipline(TorchModelPipline):
    ZeRO_MAP = {
        "SGDOptimizer": torch.optim.SGD,
        "TSMSGDOptimizer": torch.optim.SGD,
        "AdamOptimizer": torch.optim.Adam,
        "TSMAdamOptimizer": torch.optim.Adam,
        "AdamWOptimizer": torch.optim.AdamW
    }
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
                 grad_accumulate: Dict = None,
                 zero: bool = False) -> None:
        super().__init__(model, post_processing, device, criterion, optimizer,
                         lr_scheduler, pretrained, amp, grad_clip, grad_accumulate)
        self.zero = zero
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
        dist.init_process_group("nccl", init_method="tcp://" + os.environ['MASTER_ADDR'] + ":" + os.environ['MASTER_PORT'], rank=self.local_rank, world_size=self.world_size)
        torch.cuda.set_device(self.local_rank)
        if self.optimizer is not None and zero:
            self.optimizer = ZeroRedundancyOptimizer(
                self.optimizer.param_groups,
                optimizer_class=self.ZeRO_MAP[optimizer.name]
            )

    def to(self, device):
        super().to(self.local_rank)
        if not isinstance(self.model, DistributedDataParallel):
            self.model = DistributedDataParallel(self.model, device_ids=[self.local_rank])

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
            vid=cur_vid
        )
        gather_objects = [collect_dict for _ in range(self.world_size)] # any picklable object
        output_list = [None for _ in gather_objects]
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
    
    def save(self) -> Dict:
        save_dict = {}
        save_dict['model_state_dict'] = self.model.state_dict()
        if self.optimizer is not None:
            if not self.zero:
                save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
            else:
                # Todo when hit ZeroRedundancyOptimizer how it doesn't support save
                # self.optimizer.consolidate_state_dict(0)
                # save_dict['optimizer_state_dict'] = self.optimizer.state_dict()
                pass
        return save_dict
    
    def load(self, param_dict: Dict) -> None:
        self.model.load_state_dict(param_dict['model_state_dict'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(param_dict['optimizer_state_dict'])