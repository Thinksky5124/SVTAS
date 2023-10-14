'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:35:19
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 15:02:54
Description  : Transform module
FilePath     : /SVTAS/svtas/loader/transform/transform.py
'''
import torch
import copy
import numpy as np
from typing import Dict
import torchvision.transforms as transforms
from . import transform_fn as custom_transforms
from svtas.utils import AbstractBuildFactory

class BaseTransform(object):
    """Transform config to transform function

    Args:
        transform_dict: Dict[Literl|List] config of transform
    """
    def __init__(self, transform_dict={}) -> None:
        self.transforms_pipeline_list = []
        for key, cfg in transform_dict.items():
            self._get_transformers_pipline(cfg=cfg, key=key)
    
    @property
    def default_transform_name(self):
        return 'direct_transform'

    def _get_transformers_pipline(self, cfg, key):
        if isinstance(cfg, list):
            transform_op_list = []
            for transforms_op in cfg:
                name = list(transforms_op.keys())[0]
                if list(transforms_op.values())[0] is None:
                    op = getattr(transforms, name, False)
                    if op is False:
                        op = getattr(custom_transforms, name)()
                    else:
                        op = op()
                else:
                    op = getattr(transforms, name, False)
                    if op is False:
                        op = getattr(custom_transforms, name)(**list(transforms_op.values())[0])
                    else:
                        op = op(**list(transforms_op.values())[0])
                transform_op_list.append(op)
            self.transforms_pipeline_list.append([key, dict(ops=transforms.Compose(transform_op_list), transform_name=self.default_transform_name,
                                                      convert_name = key)])
        else:
            for convert_name, convert_cfg in cfg.items():
                transform_name = convert_cfg['name']
                transforms_op_list = convert_cfg['transforms_op_list']
                transforms_pipeline_dict = {}
                transforms_pipeline_dict['convert_name'] = convert_name
                transform_op_list = []
                for transforms_op in transforms_op_list:
                    name = list(transforms_op.keys())[0]
                    if list(transforms_op.values())[0] is None:
                        op = getattr(transforms, name, False)
                        if op is False:
                            op = getattr(custom_transforms, name)()
                        else:
                            op = op()
                    else:
                        op = getattr(transforms, name, False)
                        if op is False:
                            op = getattr(custom_transforms, name)(**list(transforms_op.values())[0])
                        else:
                            op = op(**list(transforms_op.values())[0])
                    transform_op_list.append(op)
                transforms_pipeline_dict['ops'] = transforms.Compose(transform_op_list)
                transforms_pipeline_dict['transform_name'] = transform_name
                self.transforms_pipeline_list.append([key, transforms_pipeline_dict])
    
    def direct_fp32_transform(self, inputs, transforms_pipeline):
        """Another Class Should overwrite this method and return a output
        """
        inputs = inputs.astype(np.float32)
        outputs = transforms_pipeline(inputs).squeeze(0)
        return outputs
    
    def direct_transform(self, inputs, transforms_pipeline):
        outputs = transforms_pipeline(inputs)
        return outputs
    
    def loop_transform(self, inputs, transforms_pipeline):
        """Another Class Should overwrite this method and return a output
        """
        outputs = []
        for input in inputs:
            input = transforms_pipeline(input)
            outputs.append(input.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs
    
    def __call__(self, results: Dict):
        for (key, transforms_pipeline) in self.transforms_pipeline_list:
            transform_func = getattr(self, transforms_pipeline['transform_name'])
            output = transform_func(results[key], transforms_pipeline=transforms_pipeline['ops'])
            results.update({transforms_pipeline['convert_name']: output})
        return results

@AbstractBuildFactory.register('dataset_transform')
class FeatureStreamTransform(BaseTransform):
    @property
    def default_transform_name(self):
        return 'direct_fp32_transform'


@AbstractBuildFactory.register('dataset_transform')
class VideoTransform(BaseTransform):
    @property
    def default_transform_name(self):
        return 'loop_transform'


@AbstractBuildFactory.register('dataset_transform')
class VideoRawStoreTransform(VideoTransform):
    def __call__(self, results):
        for (key, transforms_pipeline) in self.transforms_pipeline_list:
            results["raw_" + key] = copy.deepcopy(results[key])
            transform_func = getattr(self, transforms_pipeline['transform_name'])
            output = transform_func(results[key], transforms_pipeline=transforms_pipeline['ops'])
            results.update({transforms_pipeline['convert_name']: output})
        return results

@AbstractBuildFactory.register('dataset_transform')
class FeatureRawStoreTransform(FeatureStreamTransform):
    def __call__(self, results):
        for (key, transforms_pipeline) in self.transforms_pipeline_list:
            results["raw_" + key] = copy.deepcopy(results[key])
            transform_func = getattr(self, transforms_pipeline['transform_name'])
            output = transform_func(results[key], transforms_pipeline=transforms_pipeline['ops'])
            results.update({transforms_pipeline['convert_name']: output})
        return results
        
@AbstractBuildFactory.register('dataset_transform')
class VideoClipTransform(VideoTransform):
    
    def __call__(self, results):
        for (key, transforms_pipeline) in self.transforms_pipeline_list:
            outputs = []
            for input in results[key]:
                transform_func = getattr(self, transforms_pipeline['transform_name'])
                output = transform_func(input, transforms_pipeline=transforms_pipeline['ops'])
                outputs.append(output)
            results.update({transforms_pipeline['convert_name']: output})
        return results