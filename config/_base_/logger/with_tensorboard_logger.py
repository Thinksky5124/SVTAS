'''
Author       : Thyssen Wen
Date         : 2023-10-15 19:10:58
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-15 19:11:14
Description  : file content
FilePath     : /SVTAS/config/_base_/logger/with_tensorboard_logger.py
'''
LOGGER_LIST = dict(
    PythonLoggingLogger = dict(
        name = "SVTAS"
    ),
    TensboardLogger = dict(
        name = "SVTAS_tensorboard"
    )
)