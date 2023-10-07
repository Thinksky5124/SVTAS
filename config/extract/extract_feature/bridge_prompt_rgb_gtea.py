'''
Author       : Thyssen Wen
Date         : 2022-10-31 10:25:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-22 17:10:09
Description  : Bridge Prompt
FilePath     : /SVTAS/config/extract/extract_feature/bridge_prompt_rgb_gtea.py
'''
_base_ = [
    '../../_base_/collater/stream_compose.py'
]

sample_rate = 1
ignore_index = -100
sliding_window = 16
clip_seg_num = 16

MODEL = dict(
    architecture = "ActionCLIP",
    pretrained = "./data/checkpoint/last_model.pt",
    is_feature_extract = True,
    image_prompt = dict(
        name = "CLIP",
        # pretrained = "./data/ViT-B-16.pt",
        embed_dim = 512,
        image_resolution = 224,
        vision_layers = 12,
        vision_width = 768,
        vision_patch_size = 16,
        context_length = 77,
        vocab_size = 49408,
        transformer_width = 512,
        transformer_heads = 8,
        transformer_layers = 12,
        joint=False,
        tsm=False,
        clip_seg_num=clip_seg_num,
        dropout = 0.,
        emb_dropout = 0.,
        need_spatial=True
    ),
    text_prompt = dict(
        name = "BridgePromptTextEncoder",
        actions_map_file_path = "./data/gtea/mapping.txt"
    ),
    fusion_neck = dict(
        name = "BridgePromptFusionEarlyhyp",
        embedding_dim=512,
        clip_seg_num=clip_seg_num,
        num_layers=6,
        cnt_max=7
    ),
    head = dict(
        name = "FeatureExtractHead",
        in_channels = 512,
        input_seg_num = clip_seg_num,
        output_seg_num = clip_seg_num,
        sample_rate = sample_rate,
        pool_space = True,
        in_format = "N*T,C,H,W",
        out_format = "NCT"
    ),
    loss = None
)

PRETRAINED = None

POSTPRECESSING = dict(
    name = "StreamFeaturePostProcessing",
    sliding_window = sliding_window,
    ignore_index = ignore_index
)

DATASET = dict(
    temporal_clip_batch_size = 3,
    video_batch_size = 4,
    num_workers = 2,
    config = dict(
        name = "RawFrameStreamSegmentationDataset",
        data_prefix = "./",
        file_path = "./data/gtea/splits/all_files.txt",
        videos_path = "./data/gtea/Videos",
        gt_path = "./data/gtea/groundTruth",
        actions_map_file_path = "./data/gtea/mapping.txt",
        dataset_type = "gtea",
        train_mode = False,
        sliding_window = sliding_window
    )
)

DATASETPIPLINE = dict(
    name = "BaseDatasetPipline",
    decode = dict(
        name = "VideoDecoder",
        backend = dict(name="DecordContainer")
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
            dict(ResizeImproved = dict(size = 256)),
            dict(CenterCrop = dict(size = 224)),
            dict(PILToTensor = None),
            dict(ToFloat = None),
            dict(NormalizeColorTo1 = None),
            dict(Normalize = dict(
                mean = [140.39158961711036 / 255.0, 108.18022223151027 / 255.0, 45.72351736766547 / 255.0],
                std = [33.94421369129452 / 255.0, 35.93603536756186 / 255.0, 31.508484434367805 / 255.0]
            ))
        ])
    )
)