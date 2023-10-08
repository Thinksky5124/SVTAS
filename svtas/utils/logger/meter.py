'''
Author       : Thyssen Wen
Date         : 2023-09-24 21:16:03
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 14:15:25
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/meter.py
'''
import torch

class AverageMeter(object):
    """
    Computes and stores the average and current _value
    """

    def __init__(self, name='', fmt='f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """ reset """
        self._val = 0
        self._avg = 0
        self._sum = 0
        self._count = 0

    def update(self, _val, n=1):
        """ update """
        if isinstance(_val, torch.Tensor):
            _val = _val.cpu().detach().numpy()
        self._val = _val
        self._sum += _val * n
        self._count += n

    @property
    def total(self):
        return '{self.name}__sum: {self._sum:{self.fmt}}'.format(self=self)

    @property
    def total_minute(self):
        return '{self.name}__sum: {s:{self.fmt}} min'.format(s=self._sum / 60,
                                                            self=self)

    @property
    def count(self):
        return self._count

    @property
    def val(self):
        return self._val
    
    @property
    def avg(self):
        self._avg = self._sum / self._count
        return self._avg
    
    @property
    def sum(self):
        return self._sum

    @property
    def str_avg(self):
        if self._count != 0:
            self._avg = self._sum / self._count
        else:
            self._avg = self._sum
        return '{self.name}_avg: {self._avg:{self.fmt}}'.format(
            self=self)
        
    @property
    def str_value(self):
        return '{self.name}: {self._val:{self.fmt}}'.format(self=self)