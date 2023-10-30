'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:28:23
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 15:28:27
Description  : file content
FilePath     : /SVTAS/svtas/serving/client/visualizer/opencv_visualizer.py
'''
from .base import BaseClientViusalizer
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('serving_client_visualizer')
class OpencvViusalizer(BaseClientViusalizer):
    def __init__(self) -> None:
        super().__init__()
    
    def show(self):
        return super().show()
    
    def shutdown(self):
        return super().shutdown()