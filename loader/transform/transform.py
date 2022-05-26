'''
Author       : Thyssen Wen
Date         : 2022-05-18 15:35:19
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 16:32:19
Description  : Transform module
FilePath     : /ETESVS/loader/transform/transform.py
'''
import numpy as np
import torch
import copy
import torchvision.transforms as transforms
from ..builder import TRANSFORM

@TRANSFORM.register()
class FeatureStreamTransform():
    def __init__(self, transform_list):
        transform_op_list = []
        for transforms_op in transform_list:
            name = list(transforms_op.keys())[0]
            if list(transforms_op.values())[0] is None:
                op = getattr(transforms, name)()
            else:
                op = getattr(transforms, name)(**list(transforms_op.values())[0])
            transform_op_list.append(op)
        self.feature_transforms_pipeline = transforms.Compose(transform_op_list)

    def __call__(self, results):
        feature = results['feature'].astype(np.float32)
        feature = self.feature_transforms_pipeline(feature).squeeze(0)
        results['feature'] = copy.deepcopy(feature)
        return results

@TRANSFORM.register()
class VideoStreamTransform():
    def __init__(self, transform_list):
        transform_op_list = []
        for transforms_op in transform_list:
            name = list(transforms_op.keys())[0]
            if list(transforms_op.values())[0] is None:
                op = getattr(transforms, name)()
            else:
                op = getattr(transforms, name)(**list(transforms_op.values())[0])
            transform_op_list.append(op)
        self.imgs_transforms_pipeline = transforms.Compose(transform_op_list)

    def __call__(self, results):
        imgs = []
        for img in results['imgs']:
            img = self.imgs_transforms_pipeline(img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        return results

@TRANSFORM.register()       
class RGBFlowVideoStreamTransform():
    def __init__(self, rgb, flow):
        self.imgs_transforms_pipeline_dict = {}
        # rgb
        transform_op_list = []
        for transforms_op in rgb:
            name = list(transforms_op.keys())[0]
            if list(transforms_op.values())[0] is None:
                op = getattr(transforms, name)()
            else:
                op = getattr(transforms, name)(**list(transforms_op.values())[0])
            transform_op_list.append(op)
        self.imgs_transforms_pipeline_dict['rgb'] = transforms.Compose(transform_op_list)
        # flow
        transform_op_list = []
        for transforms_op in flow:
            name = list(transforms_op.keys())[0]
            if list(transforms_op.values())[0] is None:
                op = getattr(transforms, name)()
            else:
                op = getattr(transforms, name)(**list(transforms_op.values())[0])
            transform_op_list.append(op)
        self.imgs_transforms_pipeline_dict['flow'] = transforms.Compose(transform_op_list)

    def __call__(self, results):
        # rgb
        imgs = []
        for img in results['imgs']:
            img = self.imgs_transforms_pipeline_dict['rgb'](img)
            imgs.append(img.unsqueeze(0))
        imgs = torch.cat(imgs, dim=0)
        results['imgs'] = copy.deepcopy(imgs)
        # flow
        flows = []
        for flow in results['flows']:
            flow = self.imgs_transforms_pipeline_dict['flow'](flow)
            flows.append(flow.unsqueeze(0))
        flows = torch.cat(flows, dim=0)
        results['flows'] = copy.deepcopy(flows)
        return results
        