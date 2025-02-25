'''
Author       : Thyssen Wen
Date         : 2022-06-13 16:22:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-06 20:37:46
Description  : Stream Segmentation 2D without backbone loss
FilePath     : /ETESVS/model/architectures/segmentation/stream_segmentation2d.py
'''
import torch
import torch.nn as nn
from mmcv.runner import load_checkpoint

from utils.logger import get_logger

from ...builder import build_backbone
from ...builder import build_neck
from ...builder import build_head

from ...builder import ARCHITECTURE
from model.backbones.image.resnet import ResNet


@ARCHITECTURE.register()
class StreamSegmentation2D(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        # backbone['pretrained'] = 'data/cleaned.pth'
        # self.det_backbone = build_backbone(backbone)

        self.det_weights = torch.load('data/ew_cleaned.pth')
        self.det_backbone =  ResNet(depth=50, pretrained='data/new_cleaned.pth')
        # self.det_backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
        # self.det_weights = {key.replace('backbone.', ''):val for key, val in torch.load('data/last_epoch_model.pth').items() if key.startswith('backbone.')}
        # self.det_weights = torch.load('efficientEgoNet_bb_512x512.pth', map_location='cpu')
        # self.det_backbone = torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)
        # self.det_backbone =  torch.nn.Sequential(*(list(self.det_backbone.children())[:-2]))


        self.neck = build_neck(neck)
        self.head = build_head(head)

        # self.backbone.init_weights()
        # self.det_backbone.init_weights()

        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(in_channels=4096, out_channels=2048, kernel_size=3, padding=1, dilation=1),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, dilation=1)
        )

        self.init_weights()
        self.sample_rate = head.sample_rate

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.det_backbone is not None:
            # self.det_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r''), (r'conv.net', 'conv')])
            self.det_backbone.load_state_dict(self.det_weights, strict=True)
            print('det backbone loaded')
            for param in self.det_backbone.parameters(): 
                param.requires_grad=False
            print('det backbone freezed')

        if self.neck is not None:
            self.neck.init_weights()
        if self.head is not None:
            self.head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.head is not None:
            self.head._clear_memory_buffer()
    
    def feature_fusion(self, temporal_fet, spatial_feat):

        x_1 = temporal_fet
        x_2 = spatial_feat
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_3 = self.feat_combine(x_concat)
        feature = self.feat_refine(x_3)

        return feature

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
        import pdb; pdb.set_trace()
        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]
        if self.backbone is not None:
             # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
            feature_det =self.det_backbone(imgs, backbone_masks)#self.det_backbone(imgs).last_hidden_state # 
            
            # feature_det = self.det_backbone(imgs)

        else:
            feature = imgs
        
        feature = self.feature_fusion(feature, feature_det)
        # feature = 0.5*feature+0.5*feature_det
        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]
        if self.head is not None:
            head_score = self.head(seg_feature, masks)
        else:
            head_score = seg_feature
        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]
        return head_score
    

@ARCHITECTURE.register()
class StreamSegmentation2DMultiLabel(nn.Module):
    def __init__(self,
                 backbone=None,
                 neck=None,
                 head=None,
                 loss=None):
        super().__init__()
        self.backbone = build_backbone(backbone)
        # backbone['pretrained'] = 'data/cleaned.pth'
        # self.det_backbone = build_backbone(backbone)

        self.det_weights = torch.load('data/new_cleaned.pth')
        self.det_backbone =  ResNet(depth=50, pretrained='data/new_cleaned.pth')
        # self.det_backbone = ResNetModel.from_pretrained("microsoft/resnet-50")
        # self.det_weights = {key.replace('backbone.', ''):val for key, val in torch.load('data/last_epoch_model.pth').items() if key.startswith('backbone.')}
        # self.det_weights = torch.load('efficientEgoNet_bb_512x512.pth', map_location='cpu')
        # self.det_backbone = torchvision.models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights)
        # self.det_backbone =  torch.nn.Sequential(*(list(self.det_backbone.children())[:-2]))

        self.neck = build_neck(neck)
        self.action_head = build_head(head['action_head'])
        self.branch_head = build_head(head['branch_head'])

        # self.backbone.init_weights()
        # self.det_backbone.init_weights()

        self.feat_combine = torch.nn.Sequential(
            conv_bn_relu(in_channels=4096, out_channels=2048, kernel_size=3, padding=1, dilation=1),
            torch.nn.Conv2d(in_channels=2048, out_channels=2048, kernel_size=1),
        )

        self.feat_refine = torch.nn.Sequential(
            conv_bn_relu(in_channels=2048, out_channels=2048, kernel_size=3, padding=1, dilation=1)
        )

        self.init_weights()
        self.sample_rate = head.sample_rate

    def init_weights(self):
        if self.backbone is not None:
            self.backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r'')])

        if self.det_backbone is not None:
            # self.det_backbone.init_weights(child_model=False, revise_keys=[(r'backbone.', r''), (r'conv.net', 'conv')])
            self.det_backbone.load_state_dict(self.det_weights, strict=True)
            print('det backbone loaded')
            for param in self.det_backbone.parameters(): 
                param.requires_grad=False
            print('det backbone freezed')

        if self.neck is not None:
            self.neck.init_weights()
        if self.action_head is not None:
            self.action_head.init_weights()
        if self.branch_head is not None:
            self.branch_head.init_weights()
    
    def _clear_memory_buffer(self):
        if self.backbone is not None:
            self.backbone._clear_memory_buffer()
        if self.neck is not None:
            self.neck._clear_memory_buffer()
        if self.action_head is not None:
            self.action_head._clear_memory_buffer()
        if self.branch_head is not None:
            self.branch_head._clear_memory_buffer()
    def feature_fusion(self, temporal_fet, spatial_feat):

        x_1 = temporal_fet
        x_2 = spatial_feat
        x_concat = torch.cat([x_1, x_2], dim=1)
        x_3 = self.feat_combine(x_concat)
        feature = self.feat_refine(x_3)

        return feature

    def forward(self, input_data):
        masks = input_data['masks']
        imgs = input_data['imgs']
        # masks.shape=[N,T]
        masks = masks.unsqueeze(1)
        # x.shape=[N,T,C,H,W], for most commonly case
        imgs = torch.reshape(imgs, [-1] + list(imgs.shape[2:]))
        # x [N * T, C, H, W]
        if self.backbone is not None:
             # masks.shape [N * T, 1, 1, 1]
            backbone_masks = torch.reshape(masks[:, :, ::self.sample_rate], [-1]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            feature = self.backbone(imgs, backbone_masks)
            feature_det =self.det_backbone(imgs, backbone_masks)#self.det_backbone(imgs).last_hidden_state # 
            
            # feature_det = self.det_backbone(imgs)

        else:
            feature = imgs
        
        feature = self.feature_fusion(feature, feature_det)
        # feature = 0.5*feature+0.5*feature_det
        # feature [N * T , F_dim, 7, 7]
        # step 3 extract memory feature
        if self.neck is not None:
            seg_feature = self.neck(feature, masks[:, :, ::self.sample_rate])
            
        else:
            seg_feature = feature

        # step 5 segmentation
        # seg_feature [N, H_dim, T]
        # cls_feature [N, F_dim, T]

        if self.action_head is not None:
            action_head_score = self.action_head(seg_feature, masks)
        else:
            action_head_score = seg_feature

        if self.branch_head is not None:
            branch_head_score = self.branch_head(seg_feature, masks)
        else:
            branch_head_score = seg_feature

        # seg_score [stage_num, N, C, T]
        # cls_score [N, C, T]

        return {'branch_score':branch_head_score, 'action_score':action_head_score}