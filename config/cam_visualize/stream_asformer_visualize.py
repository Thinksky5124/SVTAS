'''
Author       : Thyssen Wen
Date         : 2022-12-23 20:48:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-20 11:57:29
Description  : file content
FilePath     : /SVTAS/config/cam_visualize/stream_asformer_visualize.py
'''
_base_ = [
    '../_base_/models/temporal_action_segmentation/asformer.py',
    '../_base_/default_runtime.py', '../_base_/collater/stream_compose.py'
]

split = 1
num_classes = 19
sample_rate = 8
clip_seg_num = 128
ignore_index = -100
epochs = 50
batch_size = 1
sliding_window = clip_seg_num * sample_rate

MODEL = dict(
    head = dict(
        num_decoders = 3,
        num_layers = 10,
        r1 = 2,
        r2 = 2,
        num_f_maps = 64,
        input_dim = 2048,
        channel_masking_rate = 0.5,
        num_classes = num_classes,
        sample_rate = sample_rate
    ),
    loss = dict(
        num_classes = num_classes,
        sample_rate = sample_rate,
        ignore_index = ignore_index
    )
)

PRETRAINED = "./output/Stream_Asformer_128x8_50salads_split2/Stream_Asformer_128x8_50salads_split2_best.pt"

VISUALIZE = dict(
    layer_name = ["model.head.decoders.2.layers.9.conv_1x1"],
    batch_size = batch_size,
    sample_rate = sample_rate,
    ignore_index = ignore_index,
    data_key = "feature",
    return_targets_name = dict(
        TemporalSegmentationTarget = dict(select_frame_idx_list=[44])
        # CategorySegmentationTarget = dict(category=0)
    ),
    reshape_transform = "NCT",
    label_path = "./data/50salads/mapping.txt",
    match_fn_name = "feature_batch_match_fn"
)

POSTPRECESSING = dict(
    name = "CAMImagePostProcessing",
    ignore_index = ignore_index,
    sample_rate = sample_rate,
    output_frame_size = [720, 404],
    need_label = True,
    need_split = True
)

DATASET = dict(
    temporal_clip_batch_size = batch_size,
    video_batch_size = batch_size,
    num_workers = 2,
    config = dict(
        name = "CAMFeatureStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./output/all_files.txt",
        feature_path = "./data/50salads/features",
        gt_path = "./data/50salads/groundTruth",
        actions_map_file_path = "./data/50salads/mapping.txt",
        dataset_type = "50salads",
        train_mode = True,
        sliding_window = sliding_window
    ),
)

PIPELINE = dict(
    name = "BasePipline",
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
        feature_dim_dict={"feature":2048},
        sample_mode = "uniform"
    ),
    transform = dict(
        name = "FeatureRawStoreTransform",
        transform_dict = dict(
            feature = [dict(XToTensor = None)]
        )
    )
)