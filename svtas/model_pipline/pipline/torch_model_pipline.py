'''
Author       : Thyssen Wen
Date         : 2023-09-21 19:24:52
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:51:21
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_model_pipline.py
'''
from typing import Any, Dict
from svtas.utils.logger import AverageMeter
from .base_pipline import BaseModelPipline
from svtas.utils import AbstractBuildFactory

import torch
import torch.distributed as dist

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # sum-up as the all-reduce operation
    rt /= nprocs # NOTE this is necessary, since all_reduce here do not perform average 
    return rt

@AbstractBuildFactory.register('model_pipline')
class TorchModelPipline(BaseModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 use_amp=False,
                 grad_clip=None,
                 need_grad_accumulate=True) -> None:
        super().__init__(model, post_processing, criterion, optimizer, lr_scheduler)
        self.grad_clip = grad_clip
        self.use_amp = use_amp
        self.need_grad_accumulate = need_grad_accumulate
        self.current_step_vid_list = None
    
    def pre_froward(self, data_dict):
        return data_dict
    
    def forward(self, data_dict):
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
        return outputs, input_data
    
    def caculate_loss(self, loss_dict):
        loss = loss_dict["loss"]
        loss.backward()
        if self.grad_clip is not None:
            for param_group in self.optimizer.param_groups:
                self.grad_clip(param_group['params'])

        if not self.need_grad_accumulate:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def after_forward(self, mode, end_info_dict):
        if mode == 'iter':
            return self.iter_end_after_forward(end_info_dict=end_info_dict)
        elif mode == 'batch':
            return self.batch_end_after_forward(end_info_dict=end_info_dict)
    
    @torch.no_grad()
    def iter_end_after_forward(self, end_info_dict):
        vid_list = end_info_dict['vid_list']
        sliding_num = end_info_dict['sliding_num']
        idx = end_info_dict['current_sliding_cnt']
        labels = end_info_dict['labels']
        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
            output = self.post_processing.update(score, labels, idx) / sliding_num
        return output
    
    @torch.no_grad()
    def batch_end_after_forward(self, end_info_dict):
        # get pred result
        pred_score_list, pred_cls_list, ground_truth_list = self.post_processing.output()
        outputs = dict(predict=pred_cls_list,
                       output_np=pred_score_list)
        vid = self.current_step_vid_list
        output_dict = dict(
            vid=vid,
            outputs=outputs,
            ground_truth=ground_truth_list
        )
        self.current_step_vid_list = end_info_dict['vid_list']
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(end_info_dict['sliding_num'], len(end_info_dict['vid_list']))
        return output_dict
    
    def caculate_loss(self, outputs, input_data) -> dict:
        loss_dict = self.criterion(outputs, input_data)
        score = outputs['output']
        return score, loss_dict
    
    def pre_backward(self, loss_dict):
        return super().pre_backward(loss_dict)
    
    def backward(self, loss_dict):
        loss = loss_dict["loss"]
        loss.backward()
        if self.grad_clip is not None:
            for param_group in self.optimizer.param_groups:
                self.grad_clip(param_group['params'])
    
    def after_backward(self, loss_dict):
        return super().after_backward(loss_dict)
    
    def memory_clear(self):
        self.model._clear_memory_buffer()

    def update_model_param(self):
        if self.need_grad_accumulate:
            self.optimizer.step()
        self.optimizer.zero_grad()
        
        self.memory_clear()
    
    def init_model_param(self, *args, **kwargs):
        self.model.init_weights()
    
    def resert_model_pipline(self, *args, **kwargs):
        return super().resert_model_pipline(*args, **kwargs)
    
    def end_model_pipline(self):
        return super().end_model_pipline()
    
    def save(self, path):
        pass

    def load(self, param_dict: Dict) -> None:
        pass

    def __call__(self, data_dict) -> Any:
        self.forward(data_dict)

@AbstractBuildFactory.register('model_pipline')
class TorchDDPModelPipline(TorchModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 criterion=None,
                 optimizer=None,
                 use_amp=False,
                 grad_clip=None,
                 need_grad_accumulate=True,
                 nprocs=1,
                 local_rank=-1) -> None:
        self.nprocs = nprocs
        self.local_rank = local_rank
        super().__init__(model=model,
                         post_processing=post_processing,
                         criterion=criterion,
                         optimizer=optimizer,
                         use_amp=use_amp,
                         grad_clip=grad_clip,
                         need_grad_accumulate=need_grad_accumulate)

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