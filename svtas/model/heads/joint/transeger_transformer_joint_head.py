'''
Author       : Thyssen Wen
Date         : 2022-06-05 10:35:39
LastEditors  : Thyssen Wen
LastEditTime : 2022-06-12 17:12:15
Description  : Transeger temporal Transformer network joint network module
FilePath     : /ETESVS/model/heads/joint/transeger_transformer_joint_head.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import (Decoder, DecoderLayer, FullAttention, ProbAttention, AttentionLayer,
                    TransformerModel, FixedPositionalEncoding, LearnedPositionalEncoding)

from ...builder import HEADS

@HEADS.register()
class TransegerTransformerJointNet(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 clip_seg_num=32,
                 num_layers=3,
                 dropout_rate=0.1,
                 encoder_heads_num=8,
                 hidden_channels=128,
                 sample_rate=4,
                 positional_encoding_type="learned"):
        super().__init__()
        self.sample_rate = sample_rate
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, 1, attention_dropout=dropout_rate),  # True
                                   in_channels, encoder_heads_num),  # ProbAttention  FullAttention
                    AttentionLayer(FullAttention(False, 1, attention_dropout=dropout_rate),  # False
                                   in_channels, encoder_heads_num),
                    in_channels,
                    hidden_channels,
                    dropout=dropout_rate,
                    activation='gelu',
                )
                for l in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(in_channels)
        )
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                clip_seg_num, in_channels, clip_seg_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                in_channels,
            )
        self.after_dropout = nn.Dropout(p=dropout_rate)
        self.conv_cls = nn.Conv1d(in_channels, num_classes, 1)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, img_feature, text_feature, masks):
        # img_feature [N D T]
        # text_feature [N D T]
        # masks [N T]

        masks = masks.unsqueeze(1)[:, :, ::self.sample_rate]

        # joint branch
        # [N D T] -> [N T D]
        img_feature = torch.permute(img_feature, dims=[0, 2, 1])
        text_feature = torch.permute(text_feature, dims=[0, 2, 1])
        img_feature = self.decoder_position_encoding(img_feature)
        # [N T D]
        output_feature = self.decoder(img_feature, text_feature)
        # [N T D] -> [N D T]
        output_feature = torch.permute(output_feature, dims=[0, 2, 1])
        output_feature = self.after_dropout(output_feature)

        # [N C T]
        joint_score = self.conv_cls(output_feature) * masks[:, 0:1, :]

        # [N C T] -> [num_satge N C T]
        outputs = joint_score.unsqueeze(0)
        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        return outputs