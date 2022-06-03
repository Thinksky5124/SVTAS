'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:01:11
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-03 11:07:42
Description  : Text backbone model
FilePath     : /ETESVS/model/backbones/language/__init__.py
'''
from .transducer_text_encoder import TransducerTextEncoder
from .cocoop_prompt import COCOOPTextEncoder
from .coop_prompt import COOPTextEncoder

__all__ = ["TransducerTextEncoder", "COCOOPTextEncoder", "COOPTextEncoder"]