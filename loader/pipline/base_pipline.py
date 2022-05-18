'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:42:16
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 15:43:32
Description  : Base Pipline class
FilePath     : /ETESVS/loader/pipline/base_pipline.py
'''
from ..builder import build_decode, build_sampler, build_transform
from ..builder import PIPLINE

@PIPLINE.register()
class BasePipline():
    def __init__(self,
                 decode=None,
                 sample=None,
                 transform=None):
        self.decode = build_decode(decode)
        self.sample = build_sampler(sample)
        self.transform = build_transform(transform)

    def __call__(self, results):
        # decode
        results = self.decode(results)
        # sample
        results = self.sample(results)
        # transform
        results = self.transform(results)
        return results