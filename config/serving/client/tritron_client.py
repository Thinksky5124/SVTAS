'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:36:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 16:44:40
Description  : file content
FilePath     : /SVTAS/config/serving/client/tritron_client.py
'''
model_name = "swinv2_onnx"

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
        server_url = "localhost:8001",
        protocol = "grpc"
    ),
    dataloader = dict(
        name = "RandomTensorNumpyDataloader",
        iter_num = 10,
        is_train = False,
        tensor_dict = dict(
            imgs = dict(
                shape = [1, 64, 3, 256, 256],
                dtype = "float32"),
            masks = dict(
                shape = [1, 64 * 2],
                dtype = "float32")
        )
    ),
    post_processing = dict(
        name = "InferencePostProcessing"
    )
)