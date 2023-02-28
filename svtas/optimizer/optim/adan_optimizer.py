'''
Author       : Thyssen Wen
Date         : 2022-10-27 10:19:27
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-27 21:06:47
Description  : Adan Optimizer ref:https://github.com/sail-sg/Adan
FilePath     : /SVTAS/svtas/optimizer/optim/adan_optimizer.py
'''
""" Adan Optimizer
Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
Implementation adapted from https://github.com/sail-sg/Adan
"""

import math
from ..builder import OPTIMIZER
import torch
from .helper_function import (filter_normal_optim_params, filter_no_decay_optim_params,
                              filter_no_decay_finetuning_optim_params, filter_finetuning_optim_params)

from torch.optim import Optimizer

@OPTIMIZER.register()
class AdanOptimizer(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(
            self,
            model,
            learning_rate=1e-3,
            betas=(0.98, 0.92, 0.99),
            eps=1e-8,
            weight_decay=0.0,
            no_prox=False,
            finetuning_scale_factor=0.1,
            no_decay_key = [],
            finetuning_key = [],
            freeze_key = [],
            **kwargs
    ):
        if not 0.0 <= learning_rate:
            raise ValueError("Invalid learning rate: {}".format(learning_rate))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, no_prox=no_prox)
        params = self.get_optim_policies(model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay)
        super(AdanOptimizer, self).__init__(params, defaults)

    def get_optim_policies(self, model, finetuning_key, finetuning_scale_factor, no_decay_key, freeze_key, learning_rate, weight_decay):
        params = list(model.named_parameters())
        no_main = no_decay_key + finetuning_key

        for n, p in params:
            for nd in freeze_key:
                if nd in n:
                    p.requires_grad = False

        normal_optim_params = filter_normal_optim_params(params=params, no_main=no_main)
        no_decay_optim_params = filter_no_decay_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)
        no_decay_finetuning_optim_params = filter_no_decay_finetuning_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)
        finetuning_optim_params = filter_finetuning_optim_params(params=params, finetuning_key=finetuning_key, no_decay_key=no_decay_key)

        param_group = [
            {'params':normal_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate},
            {'params':no_decay_optim_params, 'weight_decay':0, 'lr':learning_rate},
            {'params':no_decay_finetuning_optim_params, 'weight_decay':0, 'lr':learning_rate * finetuning_scale_factor},
            {'params':finetuning_optim_params, 'weight_decay':weight_decay, 'lr':learning_rate * finetuning_scale_factor}
        ]
        return param_group
        
    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self, closure=None):
        """ Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']
            bias_correction2 = 1.0 - beta2 ** group['step']
            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['pre_grad'] = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_diff'], state['exp_avg_sq']
                grad_diff = grad - state['pre_grad']

                exp_avg.lerp_(grad, 1. - beta1)  # m_t
                exp_avg_diff.lerp_(grad_diff, 1. - beta2)  # diff_t (v)
                update = grad + beta2 * grad_diff
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1. - beta3)  # n_t

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])
                update = (exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2).div_(denom)
                if group['no_prox']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
                else:
                    p.add_(update, alpha=-group['lr'])
                    p.data.div_(1 + group['lr'] * group['weight_decay'])

                state['pre_grad'].copy_(grad)

        return loss