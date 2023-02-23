'''
Author       : Thyssen Wen
Date         : 2023-02-22 21:27:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-22 21:52:15
Description  : file content
FilePath     : /SVTAS/config/_base_/models/action_recognition/slowonly.py
'''
model = dict(
    architecture ='Recognizer3D',
    backbone = dict(
        name='ResNet3dSlowOnly',
        depth=50,
        pretrained='./data/checkpoint/slowonly_r50_4x16x1_256e_kinetics400_flow_20200704-decb8568.pth',
        pretrained2d=False,
        in_channels=2,
        lateral=False,
        conv1_kernel=(1, 7, 7),
        conv1_stride_t=1,
        pool1_stride_t=1,
        inflate=(0, 0, 1, 1),
        norm_eval=False,
        with_pool2=False)
)