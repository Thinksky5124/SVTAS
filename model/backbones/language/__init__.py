'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:01:11
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-27 11:01:31
Description  : Text backbone model
FilePath     : /ETESVS/model/backbones/language/__init__.py
'''
from .transducer_text_encoder import TransducerTextEncoder
from .cocoop_prompt import COCOOPTextEncoder

__all__ = ["TransducerTextEncoder", "COCOOPTextEncoder"]