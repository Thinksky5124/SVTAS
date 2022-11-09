'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:35:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-09 15:07:56
Description  : Transform module
FilePath     : /SVTAS/svtas/loader/transform/transform.py
'''
from abc import abstractclassmethod
import numpy as np
import torch
import copy
import torchvision.transforms as transforms
from . import transform_fn as custom_transforms
from ..builder import TRANSFORM

class BaseTransform(object):
    def __init__(self) -> None:
        self.transforms_pipeline_dict = {}

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
    
    @abstractclassmethod
    def __call__(self, results):
        raise NotImplementedError

@TRANSFORM.register()
class FeatureStreamTransform(BaseTransform):
    def __init__(self, transform_list):
        super().__init__()
        self._get_transformers_pipline(transform_list, 'feature')

    def __call__(self, results):
        feature = results['feature'].astype(np.float32)
        feature = self.transforms_pipeline_dict['feature'](feature).squeeze(0)
        results['feature'] = copy.deepcopy(feature)
        return results

@TRANSFORM.register()
class VideoStreamTransform(BaseTransform):
    def __init__(self, transform_list):
        super().__init__()
        self._get_transformers_pipline(transform_list, 'imgs')

    def __call__(self, results):
        imgs = []
        for img in results['imgs']:
            img = self.transforms_pipeline_dict['imgs'](img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        return results

@TRANSFORM.register()       
class RGBFlowVideoStreamTransform(BaseTransform):
    def __init__(self, rgb, flow):
        super().__init__()
        self._get_transformers_pipline(rgb, 'rgb')
        self._get_transformers_pipline(flow, 'flow')

    def __call__(self, results):
        # rgb
        imgs = []
        for img in results['imgs']:
            img = self.transforms_pipeline_dict['rgb'](img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        # flow
        flows = []
        for flow in results['flows']:
            flow = self.transforms_pipeline_dict['flow'](flow)
            flows.append(flow.unsqueeze(0))
        flows = torch.cat(flows, dim=0)
        results['flows'] = copy.deepcopy(flows)
        return results

@TRANSFORM.register()
class VideoStreamRawFrameStoreTransform(VideoStreamTransform):
    def __init__(self, transform_list):
        super().__init__(transform_list)

    def __call__(self, results):
        imgs = []
        results["raw_imgs"] = copy.deepcopy(results["imgs"])
        for img in results['imgs']:
            img = self.transforms_pipeline_dict['imgs'](img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        return results

@TRANSFORM.register()       
class CompressedVideoStreamTransform(BaseTransform):
    def __init__(self, rgb, flow, res):
        super().__init__()
        self._get_transformers_pipline(rgb, 'rgb')
        self._get_transformers_pipline(flow, 'flow')
        self._get_transformers_pipline(res, 'res')

    def __call__(self, results):
        # rgb
        imgs = []
        for img in results['imgs']:
            img = self.transforms_pipeline_dict['rgb'](img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        # flow
        flows = []
        for flow in results['flows']:
            flow = self.transforms_pipeline_dict['flow'](flow)
            flows.append(flow.unsqueeze(0))
        flows = torch.cat(flows, dim=0)
        results['flows'] = copy.deepcopy(flows)
        # res
        res = []
        for res_img in results['res']:
            res_img = self.transforms_pipeline_dict['res'](res_img)
            res.append(res_img.unsqueeze(0))
        res = torch.cat(res, dim=0)
        results['res'] = copy.deepcopy(res)
        return results