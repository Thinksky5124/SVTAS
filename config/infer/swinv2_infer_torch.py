'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:27:05
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 16:55:18
Description  : file content
FilePath     : /SVTAS/config/infer/swinv2_infer_torch.py
'''
_base_ = [
    '../_base_/logger/python_logger.py',
]

num_classes = 11
sample_rate = 2
clip_seg_num = 64
ignore_index = -100
sliding_window = clip_seg_num * sample_rate
split = 1
batch_size = 1
epochs = 1

model_name = "SVTAS-RL_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea_split" + str(split)

ENGINE = dict(
    name = "StandaloneInferEngine",
    record = dict(
        name = "StreamValueRecord",
        addition_record = [],
        accumulate_type = {}
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = epochs,
        batch_size = batch_size,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchInferModelPipline",
    compile_cfg = dict(
        enable = True
    ),
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
    ),
    post_processing = dict(
        name = "StreamScorePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
)

METRIC = dict(
    TAS = dict(
        name = "TASegmentationMetric",
        overlap = [.1, .25, .5],
        actions_map_file_path = "./data/gtea/mapping.txt",
        file_output = False,
        score_output = False),
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    batch_size = batch_size,
    num_workers = 2
)

COLLATE = dict(
    infer = dict(
        name = "BatchCompose",
        to_tensor_keys = ["imgs", "flows", "res", "feature", "labels", "masks", "precise_sliding_num"],
        dropout_keys = [""],
        compress_keys = ["sliding_num", "current_sliding_cnt", "step"],
        ignore_index = ignore_index
    )
)

DATASET = dict(
    infer = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/test.split1.bundle",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = 64
    )
)

DATASETPIPLINE = dict(
    infer = dict(
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
            transform_results_list = [
                dict(DropResultsByKeyName = dict(drop_keys_list=[
                    "filename", "raw_labels", "sample_sliding_idx", "format", "frames", "frames_len", "feature_len", "video_len"
                ])),
                dict(RenameResultTransform = dict(rename_pair_dict=dict(
                    video_name = "vid_list"
                )))
            ],
            transform_key_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 256)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))],
                masks = dict(
                    masks = dict(name = 'direct_transform',
                                 transforms_op_list = [
                                     dict(NumpyDataTypeTransform = dict(dtype = "float32"))])
                )
            )
        )
    )
)