'''
Author       : Thyssen Wen
Date         : 2023-09-24 21:16:03
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-24 21:16:11
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/metric.py
'''
import torch

class AverageMeter(object):
    """
    Computes and stores the average and current value
    """

    def __init__(self, name='', fmt='f', need_avg=True, output_mean=False):
        self.name = name
        self.fmt = fmt
        self.need_avg = need_avg
        self.output_mean = output_mean
        self.reset()

    def reset(self):
        """ reset """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ update """
        if isinstance(val, torch.Tensor):
            val = val.cpu().detach().numpy()
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def total(self):
        return '{self.name}_sum: {self.sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}_sum: {s:{self.fmt}} min'.format(s=self.sum / 60,
                                                            self=self)

    @property
    def get_mean(self):
        self.avg = self.sum / self.count
        return self.avg if self.need_avg else self.val
    
    @property
    def get_sum(self):
        return self.sum

    @property
    def mean(self):
        self.avg = self.sum / self.count
        return '{self.name}_avg: {self.avg:{self.fmt}}'.format(
            self=self) if self.need_avg else ''

    @property
    def value(self):
        if self.output_mean:
            self.avg = self.sum / self.count
            return '{self.name}: {self.avg:{self.fmt}}'.format(self=self)
        return '{self.name}: {self.val:{self.fmt}}'.format(self=self)