'''
Author       : Thyssen Wen
Date         : 2023-09-24 10:48:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:25:55
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/base_logger.py
'''
import os
import abc
from enum import Enum, auto
from svtas.utils.build import AbstractBuildFactory

class LoggerLevel(Enum):
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

LOGGER_DICT = dict()

class BaseLogger:
    def __init__(self,
                 name: str,
                 root_path: str = None,
                 level=LoggerLevel.INFO) -> None:
        self.name = name
        if root_path is None:
            self.path = os.environ['SVTAS_LOG_DIR']
        else:
            self.path = root_path
        self._level = level

    def __new__(cls, name: str, root_path: str = None, level=LoggerLevel.INFO):
        if name in LOGGER_DICT.keys():
            return LOGGER_DICT[name]
        else:
            logger_instance = super().__new__(cls)
            LOGGER_DICT[name] = logger_instance
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
    def close(self):
        pass

def get_logger(name: str) -> BaseLogger:
    assert name in LOGGER_DICT.keys(), f"The log with the name of {name} was not initialized!"
    return LOGGER_DICT[name]

def setup_logger(cfg):
    for logger_class, logger_cfg in cfg.items():
        logger_cfg['logger_class'] = logger_class
        AbstractBuildFactory.create_factory('logger').create(logger_cfg, key="logger_class")