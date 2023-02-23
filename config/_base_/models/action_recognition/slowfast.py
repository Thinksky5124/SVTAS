'''
Author       : Thyssen Wen
Date         : 2023-02-22 21:27:16
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-22 21:30:03
Description  : file content
FilePath     : /SVTAS/config/_base_/models/action_recognition/slowfast.py
'''
model = dict(
    architecture='Recognizer3D',
    backbone=dict(
        name='ResNet3dSlowFast',
        pretrained=None,
        resample_rate=8,  # tau
        speed_ratio=8,  # alpha
        channel_ratio=8,  # beta_inv
        slow_pathway=dict(
            name='resnet3d',
            depth=50,
            pretrained=None,
            lateral=True,
            conv1_kernel=(1, 7, 7),
            dilations=(1, 1, 1, 1),
            conv1_stride_t=1,
            pool1_stride_t=1,
            inflate=(0, 0, 1, 1),
            norm_eval=False),
        fast_pathway=dict(
            name='resnet3d',
            depth=50,
            pretrained=None,
            lateral=False,
            base_channels=8,
            conv1_kernel=(5, 7, 7),
            conv1_stride_t=1,
            pool1_stride_t=1,
            norm_eval=False))
)