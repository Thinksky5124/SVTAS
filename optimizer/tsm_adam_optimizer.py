'''
Author       : Thyssen Wen
Date         : 2022-05-06 15:19:56
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-13 19:49:46
Description  : TSM Adam optimizer
FilePath     : /ETESVS/optimizer/tsm_adam_optimizer.py
'''
from .builder import OPTIMIZER
import torch
from mmcv.utils import SyncBatchNorm, _BatchNorm

@OPTIMIZER.register()
class TSMAdamOptimizer(torch.optim.Adam):
    def __init__(self,
                 model,
                 fc_lr5=True,
                 learning_rate=0.01,
                 betas=(0.9, 0.999),
                 weight_decay=1e-4) -> None:
        self.paramwise_cfg = dict(fc_lr5=fc_lr5)
        params = self.get_optim_policies(model)
        super().__init__(params=params, lr=learning_rate, betas=betas, weight_decay=weight_decay)
    
    def get_optim_policies(self, model):
        # use fc_lr5 to determine whether to specify higher multi-factor
        # for fc layer weights and bias.
        fc_lr5 = self.paramwise_cfg['fc_lr5']
        first_conv_weight = []
        first_conv_bias = []
        normal_weight = []
        normal_bias = []
        lr5_weight = []
        lr10_bias = []
        bn = []
        custom_ops = []
        freeze_ops = []

        conv_cnt = 0
        for m in model.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv3d):
                ps = list(m.parameters())
                conv_cnt += 1
                if conv_cnt == 1:
                    if ps[0].requires_grad is True:
                        first_conv_weight.append(ps[0])
                    else:
                        freeze_ops.append(ps[0])
                    if len(ps) == 2:
                        if ps[1].requires_grad:
                            first_conv_bias.append(ps[1])
                        else:
                            freeze_ops.append(ps[1])
                else:
                    if ps[0].requires_grad:
                        normal_weight.append(ps[0])
                    else:
                        freeze_ops.append(ps[0])
                    if len(ps) == 2:
                        if ps[1].requires_grad:
                            normal_bias.append(ps[1])
                        else:
                            freeze_ops.append(ps[1])
            elif isinstance(m, torch.nn.Linear):
                m_params = list(m.parameters())
                if m_params[0].requires_grad:
                    normal_weight.append(m_params[0])
                else:
                    freeze_ops.append(m_params[0])
                if len(m_params) == 2:
                    if m_params[1].requires_grad:
                        normal_bias.append(m_params[1])
                    else:
                        freeze_ops.append(m_params[1])
            elif isinstance(m, (_BatchNorm, SyncBatchNorm, torch.nn.GroupNorm, torch.nn.BatchNorm1d,
                                torch.nn.BatchNorm2d, torch.nn.BatchNorm3d, torch.nn.LayerNorm)):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
            elif isinstance(m, torch.nn.LSTM):
                for param in list(m.parameters()):
                    if param.requires_grad:
                        bn.append(param)
                        
            elif len(m._modules) == 0:
                if len(list(m.parameters())) > 0:
                    import pdb; pdb.set_trace()

                    raise ValueError("New atomic module type: {}. Need to give it a learning policy".format(type(m)))

        # pop the cls_head fc layer params
        last_fc_weight = normal_weight.pop()
        last_fc_bias = normal_bias.pop()
        if fc_lr5:
            lr5_weight.append(last_fc_weight)
            lr10_bias.append(last_fc_bias)
        else:
            normal_weight.append(last_fc_weight)
            normal_bias.append(last_fc_bias)

        return [
            {'params': first_conv_weight, 'lr_mult': 5 if model.backbone.modality == 'Flow' else 1, 'decay_mult': 1,
             'name': "first_conv_weight"},
            {'params': first_conv_bias, 'lr_mult': 10 if model.backbone.modality == 'Flow' else 2, 'decay_mult': 0,
             'name': "first_conv_bias"},
            {'params': normal_weight, 'lr_mult': 1, 'decay_mult': 1,
             'name': "normal_weight"},
            {'params': normal_bias, 'lr_mult': 2, 'decay_mult': 0,
             'name': "normal_bias"},
            {'params': bn, 'lr_mult': 1, 'decay_mult': 0,
             'name': "BN scale/shift"},
            {'params': custom_ops, 'lr_mult': 1, 'decay_mult': 1,
             'name': "custom_ops"},
            {'params': freeze_ops, 'lr_mult': 0, 'decay_mult': 0,
             'name': "freeze_ops"},
            # for fc
            {'params': lr5_weight, 'lr_mult': 5, 'decay_mult': 1,
             'name': "lr5_weight"},
            {'params': lr10_bias, 'lr_mult': 10, 'decay_mult': 0,
             'name': "lr10_bias"}
        ]