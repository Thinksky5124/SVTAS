'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:42:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 11:12:14
Description  : Base Pipline class
FilePath     : /SVTAS/svtas/loader/pipline/base_pipline.py
'''
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('dataset_pipline')
class BasePipline():
    def __init__(self,
                 decode=None,
                 sample=None,
                 transform=None):
        self.decode = AbstractBuildFactory.create_factory('sample_decode').create(decode)
        self.sample = AbstractBuildFactory.create_factory('dataset_sample').create(sample)
        self.transform = AbstractBuildFactory.create_factory('dataset_transform').create(transform)

    def __call__(self, results):
        # decode
        results = self.decode(results)
        # sample
        results = self.sample(results)
        # transform
        results = self.transform(results)
        return results