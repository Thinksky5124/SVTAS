'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:30:51
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 23:30:54
Description  : ref: https://github.com/open-mmlab/mmengine/blob/6c5eebb823e3c9381d63fd0cd1873ed1bd9ee9de/mmengine/fileio/handlers/yaml_handler.py
FilePath     : /SVTAS/svtas/utils/fileio/handlers/yaml_handler.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import yaml

try:
    from yaml import CDumper as Dumper  # type: ignore
    from yaml import CLoader as Loader  # type: ignore
except ImportError:
    from yaml import Loader, Dumper  # type: ignore

from .base import BaseFileHandler  # isort:skip


class YamlHandler(BaseFileHandler):

    def load_from_fileobj(self, file, **kwargs):
        kwargs.setdefault('Loader', Loader)
        return yaml.load(file, **kwargs)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        yaml.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('Dumper', Dumper)
        return yaml.dump(obj, **kwargs)