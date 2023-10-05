'''
Author       : Thyssen Wen
Date         : 2023-02-08 20:31:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 20:53:35
Description  : file content
FilePath     : /SVTAS/config/svtas/feature/asrf_50salads.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/asrf.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/50salads/50salads_stream_feature.py'
]

split = 1
num_classes = 19
sample_rate = 2
ignore_index = -100
epochs = 50
clip_seg_num = 256
sliding_window = clip_seg_num * sample_rate
dim = 2048
batch_size = 1
model_name = "Stream_ASRF_"+str(clip_seg_num)+"x"+str(sample_rate)+"_50salads_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone=None,
    neck=None,
    head = dict(
        name = "ActionSegmentRefinementFramework",
        in_channel = dim,
        num_features = 64,
        num_stages = 4,
        num_layers = 10,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        name = "ASRFLoss",
        class_weight = [0.40501603,1.7388232,0.5236841,2.3680801,0.52725035,1.8183347,
                        2.1976302,1.0866599,0.9076069,1.8409629,1.1957755,0.403674,
                        0.5133538,1.5752678,1.1706547,1.0,0.7277812,0.8284057,0.48404875],
        pos_weight = [578.1731731731732],
        num_classes = num_classes,
        sample_rate = sample_rate,
        ignore_index = -100
    )
)

POSTPRECESSING = dict(
    name = "StreamScorePostProcessingWithRefine",
    sliding_window = sliding_window,
    ignore_index = ignore_index,
    refine_method_cfg = dict(
        name = "ASRFRefineMethod",
        refinement_method="refinement_with_boundary",
        boundary_threshold=0.5,
        theta_t=15,
        kernel_size=15
    )
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/50salads/splits/train.split" + str(split) + ".bundle",
        # feature_path = './data/50salads/raw_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/50salads/flow_features"
    ),
    test = dict(
        file_path = "./data/50salads/splits/test.split" + str(split) + ".bundle",
        # feature_path = './data/50salads/raw_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/50salads/flow_features"
    )
)

LRSCHEDULER = dict(
    step_size = [epochs]
)

OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.0005,
    weight_decay = 0.01,
    betas = (0.9, 0.999)
)

PIPELINE = dict(
    train = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name='FeatureDecoder',
            backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=-1,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )
        ),
        sample = dict(
            name = "FeatureStreamSampler",
            is_train = True,
            sample_rate_dict={"feature":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num, "labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window, "labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            feature_dim_dict={"feature":dim},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    ),
    test = dict(
        name = "BaseDatasetPipline",
        decode = dict(
            name='FeatureDecoder',
            backend=dict(
                    name='NPYContainer',
                    is_transpose=False,
                    temporal_dim=-1,
                    revesive_name=[(r'(mp4|avi)', 'npy')]
                 )
        ),
        sample = dict(
            name = "FeatureStreamSampler",
            is_train = False,
            sample_rate_dict={"feature":sample_rate, "labels":sample_rate},
            clip_seg_num_dict={"feature":clip_seg_num, "labels":clip_seg_num},
            sliding_window_dict={"feature":sliding_window, "labels":sliding_window},
            sample_add_key_pair={"frames":"feature"},
            feature_dim_dict={"feature":dim},
            sample_mode = "uniform"
        ),
        transform = dict(
            name = "FeatureStreamTransform",
            transform_dict = dict(
                feature = [dict(XToTensor = None)]
            )
        )
    )
)
