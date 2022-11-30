'''
Author       : Thyssen Wen
Date         : 2022-10-28 14:46:33
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-23 15:57:34
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/mobilenetv2_breakfast.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/image_classification/mobilenet_v2.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/breakfast/breakfast_stream_video.py'
]

num_classes = 48
sample_rate = 1
clip_seg_num = 32
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 2
epochs = 50
log_interval = 100
model_name = "MobileNetV2_FC_"+str(clip_seg_num)+"x"+str(sample_rate)+"_breakfast_split" + str(split)

MODEL = dict(
    architecture = "Recognition2D",
    backbone = dict(
        name = "MobileNetV2",
        pretrained = "./data/checkpoint/mobilenet_v2_batch256_imagenet_20200708-3b2dc3af.pth",
        out_indices = (7, )
    ),
    neck = dict(
        name = "PoolNeck",
        num_classes = num_classes,
        in_channels = 1280,
        clip_seg_num = clip_seg_num,
        need_pool = True
    ),
    head = dict(
        name = "FCHead",
        num_classes = num_classes,
        sample_rate = sample_rate,
        clip_seg_num = clip_seg_num,
        drop_ratio=0.5,
        in_channels=1280
    ),
    loss = dict(
        name = "LovaszSegmentationLoss",
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
    num_workers = 2,
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
        name = "BasePipline",
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
            name = "VideoStreamTransform",
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
        name = "BasePipline",
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
            name = "VideoStreamTransform",
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
