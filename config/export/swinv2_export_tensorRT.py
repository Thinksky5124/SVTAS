'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:27:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 15:01:00
Description  : file content
FilePath     : /SVTAS/config/export/swinv2_export_tensorRT.py
'''
_base_ = [
    '../_base_/dataloader/collater/batch_compose.py',
    '../_base_/logger/python_logger.py',
]

num_classes = 11
sample_rate = 2
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 80

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

ENGINE = dict(
    name = "ExportModelEngine",
    record = dict(
        name = "ValueRecord"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    ),
    convertor = dict(
        name = "TensorRTModelConvertor",
        input_names = ['imgs', 'masks'],
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='2d',
        backbone = dict(
            name = "SwinTransformerV2",
            pretrained = "./data/checkpoint/swinv2_tiny_patch4_window8_256.pth",
            img_size=256,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=8,
            drop_path_rate=0.2,
        ), 
        neck = dict(
            name = "PoolNeck",
            in_channels = 768,
            clip_seg_num = clip_seg_num,
            need_pool = True
        ),
        head = dict(
            name = "FCHead",
            num_classes = num_classes,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            drop_ratio=0.5,
            in_channels=768
        )
    )
)

DATALOADER = dict(
    name = "RandomTensorTorchDataloader",
    iter_num = 10,
    is_train = False,
    tensor_dict = dict(
        imgs = dict(
            shape = [1, clip_seg_num, 3, 256, 256],
            dtype = "float32"),
        masks = dict(
            shape = [1, clip_seg_num * sample_rate],
            dtype = "float32")
    )
)