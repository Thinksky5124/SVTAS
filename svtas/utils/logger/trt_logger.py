'''
Author       : Thyssen Wen
Date         : 2023-10-22 18:08:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-22 18:48:34
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/trt_logger.py
'''
from ..build import AbstractBuildFactory
from .base_logger import BaseLogger, LoggerLevel, get_root_logger_instance
from ..package_utils import is_tensorrt_available
from .logging_logger import PythonLoggingLogger

if is_tensorrt_available():
    import tensorrt as trt

    @AbstractBuildFactory.register('logger')
    class TensorRTLogger(trt.ILogger, BaseLogger):
        SEVERITY_MAP = {
            trt.ILogger.Severity.INFO: LoggerLevel.INFO,
            trt.ILogger.Severity.VERBOSE: LoggerLevel.DEBUG,
            trt.ILogger.Severity.WARNING: LoggerLevel.WARNING,
            trt.ILogger.Severity.ERROR: LoggerLevel.ERROR,
            trt.ILogger.Severity.INTERNAL_ERROR: LoggerLevel.CRITICAL
        }
        python_logger: PythonLoggingLogger
        def __init__(self, level=LoggerLevel.INFO):
            trt.ILogger.__init__(self)
            self.level = level
            self.python_logger = get_root_logger_instance()
        
        def log(self, severity, msg):
            if severity is None:
                self.python_logger.log(msg=msg, level=self.SEVERITY_MAP[self.level])
            self.python_logger.log(msg=msg, level=self.SEVERITY_MAP[severity])