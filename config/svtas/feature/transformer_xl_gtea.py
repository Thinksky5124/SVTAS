'''
Author       : Thyssen Wen
Date         : 2023-02-25 19:55:04
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-26 19:32:08
Description  : file content
FilePath     : /SVTAS/config/svtas/feature/transformer_xl_gtea.py
'''
_base_ = [
    '../../_base_/schedules/optimizer/adamw.py', '../../_base_/schedules/lr/liner_step_50e.py',
    '../../_base_/default_runtime.py', '../../_base_/collater/stream_compose.py',
    '../../_base_/dataset/gtea/gtea_stream_feature.py'
]

split = 1
num_classes = 11
sample_rate = 1
ignore_index = -100
epochs = 50
clip_seg_num = 64
dim = 2048
batch_size = 1
sliding_window = clip_seg_num * sample_rate
model_name = "Stream_TransformerXL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

MODEL = dict(
    architecture = "FeatureSegmentation",
    backbone = None,
    neck = None,
    head = dict(
        name = "TransformerXL",
        in_channels=dim,
        num_classes=num_classes,
        sample_rate=sample_rate,
        n_layer=4,
        n_head=1,
        d_model=512,
        d_head=128,
        dropout=0.5,
        dropatt=0.5,
        pre_lnorm=False,
        tgt_len=8,
        ext_len=32,
        mem_len=256,  
        attn_type=0,
        clamp_len=-1
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

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2,
    train = dict(
        file_path = "./data/gtea/splits/train.split" + str(split) + ".bundle",
        feature_path = './data/gtea/raw_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/gtea/flow_features"
    ),
    test = dict(
        file_path = "./data/gtea/splits/test.split" + str(split) + ".bundle",
        feature_path = './data/gtea/raw_features',
        sliding_window = sliding_window,
        # flow_feature_path = "./data/gtea/flow_features"
    )
)

OPTIMIZER = dict(
    name = "AdamWOptimizer",
    learning_rate = 0.0005,
    weight_decay = 0.01,
    betas = (0.9, 0.999),
    need_grad_accumulate = True,
    finetuning_scale_factor=0.1,
    no_decay_key = [],
    finetuning_key = [],
    freeze_key = [],
)

LRSCHEDULER = dict(
    step_size = [epochs]
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
