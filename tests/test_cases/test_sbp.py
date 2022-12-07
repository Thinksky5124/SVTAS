'''
Author       : Thyssen Wen
Date         : 2022-11-29 21:02:06
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-07 20:12:03
Description  : Auto Test launch script use `Pytest`
FilePath     : /SVTAS/tests/test_cases/test_sbp.py
'''
import tempfile
import pytest
import os
import torch
import torch.nn as nn
import numpy as np
from logging import getLogger
from ..common.hook import TorchHook
from svtas.utils.sbp import StochasticBackPropagation
from svtas.model import backbones, necks, heads
from svtas.model.builder import BACKBONES

class Forward(nn.Module):
    def __init__(self, backbone, neck=None, head=None, sample_rate=2, use_gpu=False) -> None:
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head
        self.sample_rate = sample_rate
        self.use_gpu = use_gpu
        if use_gpu:
            self.backbone.cuda()
            self.neck.cuda()
            self.head.cuda()
    
    def forward(self, imgs, masks):
        if self.use_gpu:
            imgs = imgs.cuda()
            masks = masks.cuda()
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)

        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]

        if self.backbone is not None:
                # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
        else:
            feature = imgs

        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(
                feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        return head_score

class TestSBP:
    def genrate_model(self,
                      num_classes = 11,
                      clip_seg_num = 32,
                      sample_rate = 1):
        sbp = StochasticBackPropagation(0.5)
        SBPModule = sbp(BACKBONES.get("MobileNetV2"))
        backbone = SBPModule(pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
                    out_indices = (7, ))
        neck = necks.PoolNeck(num_classes = num_classes,
            in_channels = 1280,
            clip_seg_num = clip_seg_num,
            need_pool = True)
        head = heads.FCHead(
            num_classes = num_classes,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            drop_ratio=0.5,
            in_channels=1280
        )

        model = Forward(backbone=backbone, neck=neck, head=head, sample_rate=sample_rate, use_gpu=False)
        return model

    def genrate_model_w_hook(self,
                      num_classes = 11,
                      clip_seg_num = 32,
                      sample_rate = 1):
        @StochasticBackPropagation(0.5)
        class BackboneModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = backbones.MobileNetV2(
                    pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
                    out_indices = (7, )
                )
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        torch.nn.init.constant_(m.weight, 1.0)
                        # torch.nn.init.constant_(m.bias, 0.0)

            def forward(self, x, mask):
                x = self.linear(x, mask)
                return x
        
        backbone = BackboneModule()
        neck = necks.PoolNeck(num_classes = num_classes,
            in_channels = 1280,
            clip_seg_num = clip_seg_num,
            need_pool = True)
        head = heads.FCHead(
            num_classes = num_classes,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            drop_ratio=0.5,
            in_channels=1280
        )

        hook = TorchHook()
        hook.register_backward_hook(backbone)
        hook.register_backward_hook(neck)
        hook.register_backward_hook(head)

        model = Forward(backbone=backbone, neck=neck, head=head, sample_rate=sample_rate, use_gpu=False)
        return model, hook

    def genrate_criterion_tensors(self, hook):
        criterion_dict = dict()
        for key, values in hook.results_dict.items():
            criterion_values = []
            for v in values:
                if torch.is_tensor(v):
                    criterion_values.append(np.zeros(v.shape))
                else:
                    criterion_values.append(v)
            criterion_dict[key] = tuple(criterion_values)
        return criterion_dict

    def test_sbp_forward(self):
        num_classes = 11
        clip_seg_num = 32
        sample_rate = 1

        model, hook = self.genrate_model_w_hook(num_classes=num_classes, clip_seg_num=clip_seg_num, sample_rate=sample_rate)

        input = torch.rand((1, clip_seg_num, 3, 224, 224))
        masks = torch.ones((1, clip_seg_num * sample_rate))
        output = model(input, masks)
        hook.resert_results_dict()
        assert 1
    
    def test_sbp_backward(self):
        num_classes = 11
        clip_seg_num = 32
        sample_rate = 1

        model, hook = self.genrate_model_w_hook(num_classes=num_classes, clip_seg_num=clip_seg_num, sample_rate=sample_rate)

        input = torch.rand((1, clip_seg_num, 3, 224, 224))
        masks = torch.ones((1, clip_seg_num * sample_rate))
        output = model(input, masks)

        output.backward(torch.ones_like(output))

        criterion_dict = self.genrate_criterion_tensors(hook)

        all_close = hook.check_w_criterion(criterion_dict)
        hook.resert_results_dict()
        assert all_close, "Not all Tensor Closing!"
    
    def test_save_load_model(self):
        logger = getLogger('test')
        num_classes = 11
        clip_seg_num = 32
        sample_rate = 1

        model = self.genrate_model(num_classes=num_classes, clip_seg_num=clip_seg_num, sample_rate=sample_rate)
        tempdir = tempfile.TemporaryDirectory()
        save_path = os.path.join(tempdir.name, 'model.pt')
        checkpoint = {"model_state_dict": model.state_dict()}
        torch.save(checkpoint, save_path)
        logger.info("SBP Class save test success.")
        state_dict = torch.load(save_path)
        model.load_state_dict(state_dict["model_state_dict"])
        logger.info("SBP Class load test success.")
        assert 1