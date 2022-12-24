'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:35:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-23 21:06:40
Description  : Transform module
FilePath     : /SVTAS/svtas/loader/transform/transform.py
'''
import numpy as np
import torch
import copy
import torchvision.transforms as transforms
from . import transform_fn as custom_transforms
from ..builder import TRANSFORM

class BaseTransform(object):
    """Transform config to transform function

    Args:
        transform_dict: Dict[Literl|List] config of transform
    """
    def __init__(self, transform_dict={}) -> None:
        self.transforms_pipeline_dict = {}
        for key, cfg in transform_dict.items():
            self._get_transformers_pipline(cfg=cfg, key=key)

    def _get_transformers_pipline(self, cfg, key):
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
        self.transforms_pipeline_dict[key] = transforms.Compose(transform_op_list)
    
    @classmethod
    def transform(self, inputs, transforms_pipeline):
        """Another Class Should overwrite this method and return a output
        """
        raise NotImplementedError
    
    def __call__(self, results):
        for key, transforms_pipeline in self.transforms_pipeline_dict.items():
            output = self.transform(results[key], transforms_pipeline=transforms_pipeline)
            results[key] = output
        return results

@TRANSFORM.register()
class FeatureStreamTransform(BaseTransform):
    def transform(self, inputs, transforms_pipeline):
        inputs = inputs.astype(np.float32)
        feature = transforms_pipeline(inputs).squeeze(0)
        return feature

@TRANSFORM.register()
class VideoTransform(BaseTransform):
    def transform(self, inputs, transforms_pipeline):
        outputs = []
        for input in inputs:
            input = transforms_pipeline(input)
            outputs.append(input.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        return outputs

@TRANSFORM.register()
class VideoRawStoreTransform(VideoTransform):
    def __call__(self, results):
        for key, transforms_pipeline in self.transforms_pipeline_dict.items():
            results["raw_" + key] = copy.deepcopy(results[key])
            output = self.transform(results[key], transforms_pipeline=transforms_pipeline)
            results[key] = output
        return results

@TRANSFORM.register()
class FeatureRawStoreTransform(FeatureStreamTransform):
    def __call__(self, results):
        for key, transforms_pipeline in self.transforms_pipeline_dict.items():
            results["raw_" + key] = copy.deepcopy(results[key])
            output = self.transform(results[key], transforms_pipeline=transforms_pipeline)
            results[key] = output
        return results
        
@TRANSFORM.register()
class VideoClipTransform(VideoTransform):
    
    def __call__(self, results):
        for key, transforms_pipeline in self.transforms_pipeline_dict.items():
            outputs = []
            for input in results[key]:
                output = self.transform(input, transforms_pipeline=transforms_pipeline)
                outputs.append(output)
            results[key] = outputs
        return results