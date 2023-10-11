'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:29:02
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-10 23:29:16
Description  : https://github.com/open-mmlab/mmengine/blob/6c5eebb823e3c9381d63fd0cd1873ed1bd9ee9de/mmengine/fileio/handlers/json_handler.py
FilePath     : /SVTAS/svtas/utils/fileio/handlers/json_handler.py
'''
# Copyright (c) OpenMMLab. All rights reserved.
import json

import numpy as np

from .base import BaseFileHandler


def set_default(obj):
    """Set default json values for non-serializable values.

    It helps convert ``set``, ``range`` and ``np.ndarray`` data types to list.
    It also converts ``np.generic`` (including ``np.int32``, ``np.float32``,
    etc.) into plain numbers of plain python built-in types.
    """
    if isinstance(obj, (set, range)):
        return list(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f'{type(obj)} is unsupported for json dump')


class JsonHandler(BaseFileHandler):

    def load_from_fileobj(self, file):
        return json.load(file)

    def dump_to_fileobj(self, obj, file, **kwargs):
        kwargs.setdefault('default', set_default)
        json.dump(obj, file, **kwargs)

    def dump_to_str(self, obj, **kwargs):
        kwargs.setdefault('default', set_default)
        return json.dumps(obj, **kwargs)