'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:36:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 22:38:06
Description  : file content
FilePath     : \ETESVS\config\serving\client\tritron_client.py
'''
model_name = "swinv2_onnx"
sample_rate = 2
clip_seg_num = 8
sliding_window = clip_seg_num * sample_rate

LOGGER_LIST = dict(
    PythonLoggingLogger = dict(
        name = "SVTAS"
    )
)

CLIENT = dict(
    name = "SynchronousClient",
    connector = dict(
        name = "TritronConnector",
        model_name = "swinv2_onnx",
        server_url = "10.7.12.190:8001",
        protocol = "grpc"
    ),
    dataloader = dict(
        name = "OpencvDataloader",
        clip_seg_num = clip_seg_num,
        sample_rate = sample_rate,
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
                    dict(OpencvToPIL = None),
                    dict(ResizeImproved = dict(size = 256)),
                    dict(CenterCrop = dict(size = 256)),
                    dict(PILToTensor = None),
                    dict(ToFloat = None),
                    dict(Normalize = dict(
                        mean = [140.39158961711036, 108.18022223151027, 45.72351736766547],
                        std = [33.94421369129452, 35.93603536756186, 31.508484434367805]
                    ))]
            )
        ),
    ),
    post_processing = dict(
        name = "InferencePostProcessing"
    ),
    visualizer = dict(
        name = "OpencvViusalizer",
        label_path = "output\gtea.txt",
        clip_seg_num = clip_seg_num,
        sample_rate = sample_rate,
        vis_size = [640, 480]
    )
)