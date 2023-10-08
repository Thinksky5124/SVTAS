'''
Author       : Thyssen Wen
Date         : 2023-10-08 11:02:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 11:07:50
Description  : file content
FilePath     : /SVTAS/config/_base_/dataloader/pipline/egtea/stream_video.py
'''
sample_rate = 1
clip_seg_num = 256
sliding_window = clip_seg_num * sample_rate

DATASETPIPLINE = dict(
    train = dict(
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
            transform_dict = dict(
                imgs = [
                dict(ResizeImproved = dict(size = 256)),
                dict(RandomCrop = dict(size = 224)),
                dict(RandomHorizontalFlip = None),
                dict(PILToTensor = None),
                dict(ToFloat = None),
                dict(Normalize = dict(
                    mean = [0.47882690412518875 * 255, 0.30667687330914223 * 255, 0.1764174579795214 * 255],
                    std = [0.26380785444954574 * 255, 0.20396220265286277 * 255, 0.16305419562005563 * 255]
                ))]
            )
        )
    ),
    test = dict(
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
            transform_dict = dict(
                imgs = [
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 224)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [0.47882690412518875 * 255, 0.30667687330914223 * 255, 0.1764174579795214 * 255],
                        std = [0.26380785444954574 * 255, 0.20396220265286277 * 255, 0.16305419562005563 * 255]
                    ))]
            )
        )
    )
)