'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:46:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-24 11:17:39
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/efficientformer_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/image_classification/efficientformer.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/breakfast/breakfast_stream_video.py'
]

num_classes = 48
sample_rate = 2
clip_seg_num = 32
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 2
epochs = 50
log_interval = 100
model_name = "EfficientFormer_FC_"+str(clip_seg_num)+"x"+str(sample_rate)+"_breakfast_split" + str(split)

MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "EfficientFormer",
        arch='l1',
        pretrained="data/checkpoint/efficientformer-l1_3rdparty_in1k_20220803-d66e61df.pth",
        reshape_last_feat=True,
        drop_path_rate=0,
        init_cfg=[
            dict(
                type='TruncNormal',
                layer=['Conv2d', 'Linear'],
                std=.02,
                bias=0.),
            dict(type='Constant', layer=['GroupNorm'], val=1., bias=0.),
            dict(type='Constant', layer=['LayerScale'], val=1e-5)
        ]
    ),
    neck = dict(
       name = "PoolNeck",
        in_channels = 448,
        clip_seg_num = clip_seg_num,
        need_pool = True
    ),
    head = dict(
        name = "FCHead",
        num_classes = num_classes,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        drop_ratio=0.5,
        in_channels=448
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.0,
        ignore_index = -100
    )        
)

POSTPRECESSING = dict(
    name = "StreamScorePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

OPTIMIZER = dict(
    learning_rate = 0.0005,
    weight_decay = 1e-4,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 4,
    train = dict(
        file_path = "./data/breakfast/splits/train.split" + str(split) + ".bundle",
        sliding_window = sliding_window
    ),
    test = dict(
        file_path = "./data/breakfast/splits/test.split" + str(split) + ".bundle",
        sliding_window = sliding_window,
    )
)

PIPELINE = dict(
    train = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_list = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))
            ]
        )
    ),
    test = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name="VideoDecoder",
            backend=dict(
                    name='DecordContainer')
        ),
        sample = dict(
            name = "VideoStreamSampler",
            is_train = False,
            sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
            clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
            sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
            sample_add_key_pair={"frames":"imgs"},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "VideoTransform",
            transform_list = [
                dict(ResizeImproved = dict(size = 256)),
                dict(CenterCrop = dict(size = 224)),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.4245283568405083 * 255, 0.3904851168609079 * 255, 0.33709139617292494 * 255],
                    std = [0.26207845745959846 * 255, 0.26008439810422 * 255, 0.24623600365905168 * 255]
                ))
            ]
        )
    )
)
