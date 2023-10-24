'''
Author: Thyssen Wen
Date: 2022-04-14 15:29:53
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-24 18:31:01
Description: file content
FilePath     : /SVTAS/svtas/model/losses/__init__.py
'''
from .etesvs_loss import ETESVSLoss
from .segmentation_loss import SegmentationLoss, ActionCLIPSegmentationLoss, LSTRSegmentationLoss
from .recognition_segmentation_loss import RecognitionSegmentationLoss
from .recognition_segmentation_loss import SoftLabelRocgnitionLoss
from .steam_segmentation_loss import StreamSegmentationLoss
from .video_prediction_loss import VideoPredictionLoss
from .segmentation_clip_loss import SgementationCLIPLoss, CLIPLoss
from .bridge_prompt_clip_loss import BridgePromptCLIPLoss, BridgePromptCLIPSegmentationLoss
from .lovasz_softmax_loss import LovaszSegmentationLoss
from .dice_loss import DiceSegmentationLoss
from .asrf_loss import ASRFLoss
from .rl_dpg_loss import RLPGSegmentationLoss
from .mse_loss import DiffusionSegmentationMSELoss
from .tas_diffusion_loss import TASDiffusionStreamSegmentationLoss

__all__ = [
    'ETESVSLoss', 'SegmentationLoss', 'RecognitionSegmentationLoss',
    'StreamSegmentationLoss', 'SoftLabelRocgnitionLoss', 'VideoPredictionLoss',
    'SgementationCLIPLoss', 'CLIPLoss', 'BridgePromptCLIPLoss',
    'BridgePromptCLIPSegmentationLoss', 'ActionCLIPSegmentationLoss',
    'LSTRSegmentationLoss', 'LovaszSegmentationLoss', 'DiceSegmentationLoss',
    'ASRFLoss', 'RLPGSegmentationLoss',
    'DiffusionSegmentationMSELoss', 'TASDiffusionStreamSegmentationLoss'
]