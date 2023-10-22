'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:21:54
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 18:15:48
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/__init__.py
'''

from .base_logger import (LoggerLevel, get_logger, setup_logger, get_log_root_path,
                          BaseLogger, coloring, print_log, get_root_logger_instance)
from .logging_logger import PythonLoggingLogger
from .tensorboard_logger import TensboardLogger
from .meter import AverageMeter
from .base_record import BaseRecord
from .loss_record import StreamValueRecord, ValueRecord

# optional
try:
    from .trt_logger import TensorRTLogger
except:
    pass

__all__ = [
    "BaseLogger", "PythonLoggingLogger", "TensboardLogger", "AverageMeter",
    "BaseRecord", "StreamValueRecord", "ValueRecord", "setup_logger",
    "get_logger", 'coloring', 'print_log', 'LoggerLevel', 'get_root_logger_instance',
    'get_log_root_path', 'TensorRTLogger'
]