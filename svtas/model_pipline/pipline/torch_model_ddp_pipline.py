'''
Author       : Thyssen Wen
Date         : 2023-10-06 15:16:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-06 15:18:03
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_model_ddp_pipline.py
'''
from typing import Any, Dict
from svtas.utils.logger import AverageMeter
from svtas.utils import AbstractBuildFactory
from svtas.utils.misc import set_property
from svtas.optimizer.grad_clip import GradAccumulate, GradClip
from .torch_model_pipline import TorchModelPipline

import torch
import torch.distributed as dist

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # sum-up as the all-reduce operation
    rt /= nprocs # NOTE this is necessary, since all_reduce here do not perform average 
    return rt

@AbstractBuildFactory.register('model_pipline')
class TorchDDPModelPipline(TorchModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 amp: Dict = None,
                 grad_clip: Dict = None,
                 grad_accumulate: Dict = None) -> None:
        super().__init__(model, post_processing, criterion, optimizer,
                         lr_scheduler, amp, grad_clip, grad_accumulate)

    def memory_clear(self):
        self.model.module._clear_memory_buffer()        

    def _distribute_sync_loss_dict(self):
        for key, value in self.loss_dict.items():
            if key != "loss":
                if not isinstance(value, AverageMeter):
                    self.loss_dict[key] = reduce_mean(value, self.nprocs)
                else:
                    self.loss_dict[key].update(reduce_mean(value, self.nprocs))

    def forward(self, data_dict):
        if data_dict['current_sliding_cnt'] < data_dict['sliding_num'] - 1:
            with self.model.no_sync():
                # move data
                input_data = {}
                for key, value in data_dict.items():
                    if torch.is_tensor(value):
                        if torch.cuda.is_available():
                            input_data[key] = value.cuda()
                        else:
                            input_data[key] = value
                if not self.need_grad_accumulate:
                    input_data['precise_sliding_num'] = torch.ones_like(input_data['precise_sliding_num'])

                outputs = self.model(input_data)
                loss_dict = self.criterion(outputs, input_data)
                score = outputs['output']
        return score, loss_dict

    def after_forward(self, end_info_dict):
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
        self.current_step_vid_list = end_info_dict['vid_list']

        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(end_info_dict['sliding_num'], len(end_info_dict['vid_list']))
        
        torch.distributed.barrier()
        self._distribute_sync_loss_dict()
        return output_dict
    
@AbstractBuildFactory.register('model_pipline')
class TorchFSDPModelPipline(TorchModelPipline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)