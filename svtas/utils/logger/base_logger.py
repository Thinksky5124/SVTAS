'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:21:54
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-19 18:25:04
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/base_logger.py
'''
import os
import abc
from enum import Enum, auto
from svtas.utils.build import AbstractBuildFactory
from svtas.dist import get_rank_from_os, get_world_size_from_os

Color = {
    'RED': '\033[31m',
    'HEADER': '\033[35m',  # deep purple
    'PURPLE': '\033[95m',  # purple
    'OKBLUE': '\033[94m',
    'OKGREEN': '\033[92m',
    'WARNING': '\033[93m',
    'FAIL': '\033[91m',
    'ENDC': '\033[0m'
}


def coloring(message, color="OKGREEN"):
    assert color in Color.keys()
    if os.environ.get('COLORING', True):
        return Color[color] + str(message) + Color["ENDC"]
    else:
        return message
    
class LoggerLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

def get_log_root_path() -> str:
    return os.environ['SVTAS_LOG_DIR']

class BaseLogger:
    LOGGER_DICT = dict()
    def __init__(self,
                 name: str,
                 root_path: str = None,
                 level=LoggerLevel.INFO) -> None:
        self.name = name
        if root_path is None:
            self.path = get_log_root_path()
        else:
            self.path = root_path
        self._level = level
        self.local_rank = get_rank_from_os()
        self.world_size = get_world_size_from_os()

    def __new__(cls, name: str, root_path: str = None, level=LoggerLevel.INFO):
        if name in BaseLogger.LOGGER_DICT.keys():
            raise NameError(f"You can't build two logger with the same name: {name}!")
        else:
            logger_instance = super().__new__(cls)
            BaseLogger.LOGGER_DICT[name] = logger_instance
            return logger_instance

    @abc.abstractmethod
    def log(self, msg, *args, **kwargs):
        pass

    @abc.abstractmethod
    def info(self,
             msg: object,
             *args: object,
             **kwargs):
        pass

    @abc.abstractmethod
    def debug(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    @abc.abstractmethod
    def warn(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    @abc.abstractmethod
    def error(self,
             msg: object,
             *args: object,
             **kwargs):
        pass
    
    @abc.abstractmethod
    def critical(self,
             msg: object,
             *args: object,
             **kwargs):
        pass

    @abc.abstractmethod
    def log_epoch(self, metric_list, epoch, mode, ips):
        pass

    @abc.abstractmethod
    def log_batch(self, metric_list, batch_id, mode, ips, epoch_id=None, total_epoch=None):
        pass

    @abc.abstractmethod
    def log_step(self, metric_list, step_id, mode, ips, total_step=None):
        pass
    
    @abc.abstractmethod
    def log_metric(self, metric_dict, epoch, mode):
        pass
    
    @abc.abstractmethod
    def close(self):
        pass

def get_logger(name: str) -> BaseLogger:
    assert name in BaseLogger.LOGGER_DICT.keys(), f"The log with the name of {name} was not initialized!"
    return BaseLogger.LOGGER_DICT[name]

def setup_logger(cfg):
    for logger_class, logger_cfg in cfg.items():
        logger_cfg['logger_class'] = logger_class
        AbstractBuildFactory.create_factory('logger').create(logger_cfg, key="logger_class")

def get_root_logger_instance(default_name='SVTAS'):
    return BaseLogger.LOGGER_DICT[default_name]

def print_log(msg,
              logger: BaseLogger = None,
              level=None) -> None:
    """Print a log message.

    Args:
        msg (str): The message to be logged.
        logger (Logger or str, optional): If the type of logger is
        ``logging.Logger``, we directly use logger to log messages.
            Some special loggers are:

            - "silent": No message will be printed.
            - "current": Use latest created logger to log message.
            - other str: Instance name of logger. The corresponding logger
              will log message if it has been created, otherwise ``print_log``
              will raise a `ValueError`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object, "current", or a created logger instance name.
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, BaseLogger):
        logger.log(msg, level=level)
    elif logger == 'silent':
        pass
    elif logger == 'current':
        logger_instance = get_root_logger_instance()
        logger_instance.log(level, msg)
    else:
        raise TypeError(
            '`logger` should be either a logging.Logger object, str, '
            f'"silent", "current" or None, but got {type(logger)}')