'''
Author       : Thyssen Wen
Date         : 2023-10-10 23:21:54
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 09:50:33
Description  : file content
FilePath     : /SVTAS/svtas/utils/fileio/__init__.py
'''
from .stream_writer import (StreamWriter, ImageStreamWriter, VideoStreamWriter,
                            CAMImageStreamWriter, CAMVideoStreamWriter, NPYStreamWriter)
from .io import load, dump, get_file_backend
from .file_client import FileClient