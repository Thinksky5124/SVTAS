'''
Author       : Thyssen Wen
Date         : 2022-12-22 16:37:36
LastEditors  : Thyssen Wen
LastEditTime : 2024-01-21 16:08:52
Description  : file content
FilePath     : /SVTAS/config/extract/extract_feature/i3d_flow_gtea.py
'''
_base_ = [
    '../../_base_/dataloader/collater/batch_compose.py',
    '../../_base_/logger/python_logger.py',
]

sample_rate = 1
batch_size = 4
ignore_index = -100
sliding_window = 1
clip_seg_num = 64

model_name = "I3D_Flow_"+str(clip_seg_num)+"x"+str(sample_rate)+"_UBnormal"

ENGINE = dict(
    name = "ExtractFeatureEngine",
    out_path = "data/feature/UBnormal/test_flow_feature",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = 1,
        batch_size = batch_size,
        logger_iter_interval = 10,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)

MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    # pretrained = "",
    model = dict(
        name = "StreamVideoSegmentation",
        architecture_type ='3d',
        backbone = dict(
            name = "I3D",
            in_channels=2,
            pretrained="./data/checkpoint/i3d_flow.pt",
        ),
        neck = None,
        head = dict(
            name = "FeatureExtractHead",
            in_channels = 1024,
            input_seg_num = clip_seg_num // 8,
            output_seg_num = 1,
            sample_rate = sample_rate * 8,
            pool_space = True,
            in_format = "N,C,T,H,W",
            out_format = "NCT"
        )
    ),
    post_processing = dict(
        name = "StreamFeaturePostProcessing",
        sliding_window = sliding_window,
        ignore_index = ignore_index
    ),
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/UBnormal/all_file_list_test.txt",
        videos_path = "./data/feature/UBnormal/flow_test",
        gt_path = "./data/UBnormal/groundTruth/test",
        actions_map_file_path = "./data/UBnormal/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

DATASETPIPLINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name="DecordContainer", to_ndarray=True)
    ),
    sample = dict(
        name = "VideoStreamSampler",
        is_train = False,
        sample_rate_dict={"imgs":sample_rate,"labels":sample_rate},
        clip_seg_num_dict={"imgs":clip_seg_num ,"labels":clip_seg_num},
        sliding_window_dict={"imgs":sliding_window,"labels":sliding_window},
        sample_add_key_pair={"frames":"imgs"},
        channel_mode_dict={"imgs":"XY"},
        channel_num_dict={"imgs":2},
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
                dict(XToTensor = None),
                dict(ToFloat = None),
                dict(TensorPermute = dict(permute_list = [2, 0, 1])),
                dict(TensorImageResize = dict(size = 256)),
                dict(TensorCenterCrop = dict(crop_size = 224)),
                dict(ScaleTo1_1 = None)
        ])
    )
)