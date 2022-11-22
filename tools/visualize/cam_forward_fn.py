'''
Author       : Thyssen Wen
Date         : 2022-10-23 15:24:25
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-21 23:03:46
Description  : CAM Forwardfunction override
FilePath     : /SVTAS/tools/visualize/cam_forward_fn.py
'''
import torch
#! modify clip_seg_num = 8

def cam_forward(self, input_data):
    """
        Use Like 
        ```
        feature = debugger.debug(feature, 'backbone_output') 
        ```
        for Debug
    """
    input_data = input_data.reshape([-1, 32]+list(input_data.shape[-3:]))
    masks = torch.full([input_data.shape[0], input_data.shape[1] * self.sample_rate], 1.0).to(input_data.device)
    imgs = input_data

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
    # seg_score [stage_num, N, C, T]
    # cls_score [N, C, T]
    head_score = torch.reshape(head_score[-1].transpose(1, 2), [-1, head_score.shape[-2]])
    return head_score