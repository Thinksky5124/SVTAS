'''
Author       : Thyssen Wen
Date         : 2022-10-27 11:09:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 16:19:09
Description  : RAFT extract flow Config
FilePath     : /SVTAS/config/extract/extract_flow/raft_gtea.py
'''
_base_ = [
    '../../_base_/dataloader/collater/stream_compose.py',
    '../../_base_/logger/python_logger.py',
]
sliding_window = 32
clip_seg_num = 32
sample_rate = 1
batch_size = 1

model_name = "RAFT_"+str(clip_seg_num)+"x"+str(sample_rate)+"_gtea"

ENGINE = dict(
    name = "ExtractOpticalFlowEngine",
    out_path = "data/gtea/test/extract_flow",
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
        name = "OpticalFlowEstimation",
        model = dict(
            name = "RAFT",
            pretrained = "./data/checkpoint/raft-sintel.pth",
            extract_mode = True,
            freeze = True,
            mode = "sintel"
        )
    ),
    post_processing = dict(
        name = "OpticalFlowPostProcessing",
        fps = 15,
        need_visualize = False,
        sliding_window = sliding_window
    ),
)

DATALOADER = dict(
    name = "TorchStreamDataLoader",
    temporal_clip_batch_size = 3,
    video_batch_size = batch_size,
    num_workers = 2
)

DATASET = dict(
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = clip_seg_num
    )
)


DATASETPIPLINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name='DecordContainer')
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
        transform_dict = dict(
            imgs = [
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                    std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                ))
            ])
    )
)
