'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:33:37
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 10:13:09
Description  : ref: https://github.com/open-mmlab/mmengine/blob/6c5eebb823e3c9381d63fd0cd1873ed1bd9ee9de/mmengine/fileio/backends/base.py
FilePath     : /SVTAS/svtas/utils/fileio/backends/base.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod

from svtas.utils.logger import print_log, LoggerLevel

class BaseStorageBackend(metaclass=ABCMeta):
    """Abstract class of storage backends.

    All backends need to implement two apis: :meth:`get()` and
    :meth:`get_text()`.

    - :meth:`get()` reads the file as a byte stream.
    - :meth:`get_text()` reads the file as texts.
    """

    # a flag to indicate whether the backend can create a symlink for a file
    # This attribute will be deprecated in future.
    _allow_symlink = False

    @property
    def allow_symlink(self):
        print_log(
            'allow_symlink will be deprecated in future',
            logger='current',
            level=LoggerLevel.WARNING)
        return self._allow_symlink

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def get(self, filepath):
        pass

    @abstractmethod
    def get_text(self, filepath):
        pass
