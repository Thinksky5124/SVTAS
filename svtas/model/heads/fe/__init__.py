'''
Author       : Thyssen Wen
Date         : 2022-06-06 20:19:41
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-15 21:23:20
Description  : Feature Extract Head Modules
FilePath     : /ETESVS/model/heads/feature_extractor/__init__.py
'''
from .feature_extract_head import FeatureExtractHead
from .identity_embedding_head import IdentityEmbeddingHead

__all__ = ["FeatureExtractHead", "IdentityEmbeddingHead"]