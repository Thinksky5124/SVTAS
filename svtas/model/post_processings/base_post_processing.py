'''
Author       : Thyssen Wen
Date         : 2023-10-05 19:36:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 19:41:44
Description  : file content
FilePath     : /SVTAS/svtas/model/post_processings/base_post_processing.py
'''
import abc
from typing import List

class BasePostProcessing(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def init_scores(self, sliding_num, batch_size):
        raise NotImplementedError("You must implement init_scores function!")
    
    @abc.abstractmethod
    def update(self, seg_scores, gt, idx):
        raise NotImplementedError("You must implement update function!")
    
    @abc.abstractmethod
    def output(self) -> List:
        raise NotImplementedError("You must implement output function!")