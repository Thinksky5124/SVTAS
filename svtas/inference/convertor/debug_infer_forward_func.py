'''
Author       : Thyssen Wen
Date         : 2022-09-24 16:53:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 20:46:18
Description  : New Forward Function for infer debugging
FilePath     : /SVTAS/svtas/tasks/debug_infer_forward_func.py
'''
import torch
from ...model.debugger import Debugger

debugger = Debugger()

def infer_forward(self, input_data):
    """
        Use Like 
        ```
        feature = debugger.debug(feature, 'backbone_output') 
        ```
        for Debug
    """
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
    else:
        feature = imgs

    feature = debugger.debug(feature, 'backbone_output')

    # feature [N * T , F_dim, 7, 7]
    # step 3 extract memory feature
    if self.neck is not None:
        seg_feature = self.neck(
            feature, masks[:, :, ::self.sample_rate])
        
    else:
        seg_feature = feature

    seg_feature = debugger.debug(seg_feature, 'neck_output')

    # step 5 segmentation
    # seg_feature [N, H_dim, T]
    # cls_feature [N, F_dim, T]
    if self.head is not None:
        head_score = self.head(seg_feature, masks)
    else:
        head_score = seg_feature
    # seg_score [stage_num, N, C, T]
    # cls_score [N, C, T]
    head_score = debugger.debug(head_score, 'head_output')
    return head_score