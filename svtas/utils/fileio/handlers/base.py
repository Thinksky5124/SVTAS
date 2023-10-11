'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:28:28
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 23:28:30
Description  : ref: https://github.com/open-mmlab/mmengine/blob/6c5eebb823e3c9381d63fd0cd1873ed1bd9ee9de/mmengine/fileio/handlers/base.py
FilePath     : /SVTAS/svtas/utils/fileio/handlers/base.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod


class BaseFileHandler(metaclass=ABCMeta):
    # `str_like` is a flag to indicate whether the type of file object is
    # str-like object or bytes-like object. Pickle only processes bytes-like
    # objects but json only processes str-like object. If it is str-like
    # object, `StringIO` will be used to process the buffer.
    str_like = True

    @abstractmethod
    def load_from_fileobj(self, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_fileobj(self, obj, file, **kwargs):
        pass

    @abstractmethod
    def dump_to_str(self, obj, **kwargs):
        pass

    def load_from_path(self, filepath, mode='r', **kwargs):
        with open(filepath, mode) as f:
            return self.load_from_fileobj(f, **kwargs)

    def dump_to_path(self, obj, filepath, mode='w', **kwargs):
        with open(filepath, mode) as f:
            self.dump_to_fileobj(obj, f, **kwargs)