'''
Author       : Thyssen Wen
Date         : 2023-10-11 23:10:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 23:10:50
Description  : file content
FilePath     : /SVTAS/svtas/model/scheduler/diffact_scheduler.py
'''


from svtas.utils import AbstractBuildFactory
from .base_scheduler import BaseDiffusionScheduler

@AbstractBuildFactory.register('diffusion_scheduler')
class DiffsusionActionSegmentationScheduler(BaseDiffusionScheduler):
    def __init__(self) -> None:
        super().__init__()