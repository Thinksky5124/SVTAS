'''
Author       : Thyssen Wen
Date         : 2022-11-05 20:27:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-06 21:27:27
Description  : file content
FilePath     : /SVTAS/config/svtas/rgb/i3d_rgb_flow_asformer_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py'
]
split = 1
num_classes = 11
sample_rate = 1
clip_seg_num = 256
sliding_window = 256
ignore_index = -100
batch_size = 1
epochs = 50
model_name = "I3D_Flow_Rgb_Asformer_256x1_gtea_split" + str(split)

MODEL = dict(
    architecture = "MultiModalityStreamSegmentation",
    rgb_backbone = dict(
        name = "I3D",
        pretrained = "./data/checkpoint/i3d_rgb.pt",
        in_channels = 3
    ),
    flow_backbone = dict(
        name = "I3D",
        pretrained = "./data/checkpoint/i3d_flow.pt",
        in_channels = 2
    ),
    neck = dict(
        name = "MultiModalityFusionNeck",
        fusion_mode='stack',
        clip_seg_num = clip_seg_num // 8,
        fusion_neck_module = dict(
            name = "AvgPoolNeck",
            num_classes = num_classes,
            in_channels = 2048,
            clip_seg_num = clip_seg_num // 8,
            drop_ratio = 0.5,
            need_pool = True
        )
    ),
    head = dict(
        name = "ASFormer",
        num_decoders = 3,
        num_layers = 10,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 2048,
        channel_masking_rate = 0.5,
        num_classes = num_classes,
        sample_rate = sample_rate * 8
    ),
    loss = dict(
        name = "SegmentationLoss",
        num_classes = num_classes,
        sample_rate = sample_rate,
        smooth_weight = 0.15,
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

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = batch_size * 2,
    train = dict(
        name = "RGBFlowFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/train.split1.bundle",
        videos_path = "./data/gtea/Videos",
        flows_path = './data/gtea/flow',
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = True,
        sliding_window = sliding_window
    ),
    test = dict(
        name = "RGBFlowFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        flows_path = './data/gtea/flow',
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

METRIC = dict(
    name = "TASegmentationMetric",
    overlap = [.1, .25, .5],
    actions_map_file_path = "./data/gtea/mapping.txt",
    file_output = False,
    score_output = False
)

PIPELINE = dict(
    train = dict(
        name = "BasePipline",
        decode = dict(
            name = "RGBFlowVideoDecoder",
            rgb_backend = "decord",
            flow_backend = "numpy"
        ),
        sample = dict(
            name = "RGBFlowVideoStreamSampler",
            is_train = False,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "RGBFlowVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ]
        )
    ),
    test = dict(
        name = "BasePipline",
        decode = dict(
            name = "RGBFlowVideoDecoder",
            rgb_backend = "decord",
            flow_backend = "numpy"
        ),
        sample = dict(
            name = "RGBFlowVideoStreamSampler",
            is_train = False,
            sample_rate = sample_rate,
            clip_seg_num = clip_seg_num,
            sliding_window = sliding_window,
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "RGBFlowVideoStreamTransform",
            rgb = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ],
            flow = [
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
            ]
        )
    )
)
