'''
Author       : Thyssen Wen
Date         : 2023-09-28 19:42:01
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:48:27
Description  : file content
FilePath     : /SVTAS/svtas/loader/dataloader/base_dataloader.py
'''
import abc

class BaseDataloader(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass
    
    @abc.abstractmethod
    def shuffle_dataloader(self, epoch) -> None:
        pass