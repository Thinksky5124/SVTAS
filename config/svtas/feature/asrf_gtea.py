'''
Author       : Thyssen Wen
Date         : 2023-02-08 20:31:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-13 21:39:00
Description  : file content
FilePath     : /SVTAS/config/svtas/feature/asrf_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/models/temporal_action_segmentation/asrf.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
clip_seg_num = 256
sliding_window = clip_seg_num * sample_rate
dim = 768
batch_size = 1
model_name = "Stream_ASRF_Swin3DSBP_feature_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

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
        class_weight = [0.40253314,0.6060787,0.41817436,1.0009843,1.6168522,
                        1.2425169,1.5743035,0.8149039,7.6466165,1.0,0.29321033],
        pos_weight = [33.866594360086765],
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
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        feature_path = './data/gtea/extract_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = './data/gtea/extract_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/gtea/flow_features"
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
