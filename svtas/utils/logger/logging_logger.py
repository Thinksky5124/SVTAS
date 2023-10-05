'''
Author       : Thyssen Wen
Date         : 2023-09-24 20:37:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 20:36:09
Description  : file content
FilePath     : /SVTAS/svtas/utils/logger/logging_logger.py
'''
from ..build import AbstractBuildFactory
from .base_logger import BaseLogger, LoggerLevel
import os
import sys
import datetime
import logging

def time_zone(sec, fmt):
    real_time = datetime.datetime.now()
    return real_time.timetuple()

@AbstractBuildFactory.register('logger')
class PythonLoggingLogger(BaseLogger):
    logger: logging.Logger
    level_map = {
        LoggerLevel.DEBUG: logging.DEBUG,
        LoggerLevel.INFO: logging.INFO,
        LoggerLevel.WARNING: logging.WARNING,
        LoggerLevel.ERROR: logging.ERROR,
        LoggerLevel.CRITICAL: logging.CRITICAL}
    
    def __init__(self, name: str = "SVTAS", root_path: str = None, level=LoggerLevel.INFO) -> None:
        super().__init__(name, root_path, level)
        logging.Formatter.converter = time_zone
        self.logger = logging.getLogger(name)
        self.set_level = level
        self.logger.propagate = False
        self.logger.level = self.level

        if level == "DEBUG":
            plain_formatter = logging.Formatter(
                "[%(asctime)s] %(name)s %(levelname)s: %(message)s",
                datefmt="%m/%d %H:%M:%S")
        else:
            plain_formatter = logging.Formatter(
                "[%(asctime)s] %(message)s",
                datefmt="%m/%d %H:%M:%S")
        local_rank = int(os.environ['LOCAL_RANK'])
        self.local_rank = local_rank
        if local_rank < 0:
            local_rank = 0

        if local_rank == 0:
            # stdout logging: master only
            ch = logging.StreamHandler(stream=sys.stdout)
            ch.setLevel(self.level)
            formatter = plain_formatter
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)

        # file logging: all workers
        if self.path is not None:
            if self.path.endswith(".txt") or self.path.endswith(".log"):
                filename = self.path
            else:
                # aviod cover
                filename = os.path.join(self.path, name + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") +".log")
                if(os.path.exists(filename)):
                    filename = os.path.join(self.path, name + "_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") +".log")

            # PathManager.mkdirs(os.path.dirname(filename))
            os.makedirs(os.path.dirname(filename), exist_ok=True)

            # fh = logging.StreamHandler(_cached_log_stream(filename)
            fh = logging.FileHandler(filename, mode='a')
            fh.setLevel(self.level)
            fh.setFormatter(plain_formatter)
            self.logger.addHandler(fh)

    @property
    def level(self):
        """
        Return logger's current log level
        """
        return self.level_map[self._level]
    
    @level.setter
    def level(self, level: LoggerLevel):
        assert level in LoggerLevel, f"Unsupport logging Level: {self._level}!"
        self._level = level
        self.logger.setLevel(self.level_map[level])

    def base_dist_logging(self,
                     logging_fn,
                     msg: object,
                     *args: object,
                     exc_info = None,
                     stack_info: bool = False,
                     stacklevel: int = 1,
                     extra = None):
        final_message = "[Rank {}] {}".format(self.local_rank, msg)
        logging_fn(final_message,
                    *args,
                    exc_info = exc_info,
                    stack_info = stack_info,
                    stacklevel = stacklevel,
                    extra = extra)
        
    def base_logging(self,
                     logging_fn,
                     msg: object,
                     *args: object,
                     exc_info = None,
                     stack_info: bool = False,
                     stacklevel: int = 1,
                     extra = None):
        logging_fn(msg)
    
    def log(self,
            msg: object,
            *args: object,
            log_dist: bool = False,
            exc_info = None,
            stack_info: bool = False,
            stacklevel: int = 1,
            extra = None):
        if self._level == LoggerLevel.INFO:
            self.info(msg, *args, log_dist=log_dist, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
        elif self._level == LoggerLevel.DEBUG:
            self.debug(msg, *args, log_dist=log_dist, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
        elif self._level == LoggerLevel.WARNING:
            self.warn(msg, *args, log_dist=log_dist, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
        elif self._level == LoggerLevel.ERROR:
            self.error(msg, *args, log_dist=log_dist, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
        elif self._level == LoggerLevel.CRITICAL:
            self.critical(msg, *args, log_dist=log_dist, exc_info=exc_info, stack_info=stack_info, stacklevel=stacklevel, extra=extra)
        else:
            raise NotImplementedError(f"Unsupport logging Level: {self._level}!")
            
    def info(self,
             msg: object,
             *args: object,
             log_dist: bool = False,
             exc_info = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra = None):
        if not log_dist:
            self.base_logging(self.logger.info, msg, *args, exc_info, stack_info, stacklevel, extra)
        else:
            self.base_dist_logging(self.logger.info, msg, *args, exc_info, stack_info, stacklevel, extra)

    def debug(self,
             msg: object,
             *args: object,
             log_dist: bool = False,
             exc_info = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra = None):
        if not log_dist:
            self.base_logging(self.logger.debug, msg, *args, exc_info, stack_info, stacklevel, extra)
        else:
            self.base_dist_logging(self.logger.debug, msg, *args, exc_info, stack_info, stacklevel, extra)
    
    def warn(self,
             msg: object,
             *args: object,
             log_dist: bool = False,
             exc_info = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra = None):
        if not log_dist:
            self.base_logging(self.logger.warn, msg, *args, exc_info, stack_info, stacklevel, extra)
        else:
            self.base_dist_logging(self.logger.warn, msg, *args, exc_info, stack_info, stacklevel, extra)
    
    def error(self,
             msg: object,
             *args: object,
             log_dist: bool = False,
             exc_info = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra = None):
        if not log_dist:
            self.base_logging(self.logger.error, msg, *args, exc_info, stack_info, stacklevel, extra)
        else:
            self.base_dist_logging(self.logger.error, msg, *args, exc_info, stack_info, stacklevel, extra)
    
    def critical(self,
             msg: object,
             *args: object,
             log_dist: bool = False,
             exc_info = None,
             stack_info: bool = False,
             stacklevel: int = 1,
             extra = None):
        if not log_dist:
            self.base_logging(self.logger.critical, msg, *args, exc_info, stack_info, stacklevel, extra)
        else:
            self.base_dist_logging(self.logger.critical, msg, *args, exc_info, stack_info, stacklevel, extra)
    
    def log_epoch(self, metric_list, epoch, mode, ips):
        batch_cost = 'avg_' + str(metric_list['batch_time'].value) + ' sec,'
        reader_cost = 'avg_' + str(metric_list['reader_time'].value) + ' sec,'
        batch_sum = str(metric_list['batch_time'].total) + ' sec,'

        metric_values = []
        for m in metric_list:
            if not (m == 'batch_time' or m == 'reader_time'):
                metric_values.append(metric_list[m].mean)
        metric_str = ' '.join([str(v) for v in metric_values])

        end_epoch_str = "END epoch:{:<3d}".format(epoch)
        self.logger.info("{:s} {:s} {:s} {:s} {:s} {:s} {}".format(
            end_epoch_str, mode, metric_str, batch_cost, reader_cost, batch_sum, ips))

    def log_batch(self, metric_list, batch_id, mode, ips, epoch_id=None, total_epoch=None):
        batch_cost = str(metric_list['batch_time'].value) + ' sec,'
        reader_cost = str(metric_list['reader_time'].value) + ' sec,'

        metric_values = []
        for m in metric_list:
            if not (m == 'batch_time' or m == 'reader_time'):
                metric_values.append(metric_list[m].value)
        metric_str = ' '.join([str(v) for v in metric_values])
        if epoch_id and total_epoch:
            epoch_str = "epoch:[{:>3d}/{:<3d}]".format(epoch_id, total_epoch)
        step_str = "{:s} step:{:<4d}".format(mode, batch_id)

        if epoch_id and total_epoch:
            self.logger.info("{:s} {:s} {:s} {:s} {:s} {}".format(
                epoch_str, step_str, metric_str, batch_cost, reader_cost, ips))
        else:
            self.logger.info("{:s} {:s} {:s} {:s} {}".format(
                step_str, metric_str, batch_cost, reader_cost, ips))
    
    def log_step(self, metric_list, step_id, mode, ips, total_step=None):
        batch_cost = str(metric_list['batch_time'].value) + ' sec,'
        reader_cost = str(metric_list['reader_time'].value) + ' sec,'

        metric_values = []
        for m in metric_list:
            if not (m == 'batch_time' or m == 'reader_time'):
                metric_values.append(metric_list[m].value)
        metric_str = ' '.join([str(v) for v in metric_values])
        if total_step:
            step_str = "{:s} step:[{:>3d}/{:<3d}]".format(mode, step_id, total_step)
        else:
            step_str = "{:s} step:{:<4d}".format(mode, step_id)

        self.logger.info("{:s} {:s} {:s} {:s} {}".format(
                step_str, metric_str, batch_cost, reader_cost, ips))
    
    def close(self):
        self.logger.shutdown()