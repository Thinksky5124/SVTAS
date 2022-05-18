'''
Author       : Thyssen Wen
Date         : 2022-05-17 14:57:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 14:54:27
Description  : OADTR model ref:https://github.com/wangxiang1230/OadTR
FilePath     : /ETESVS/model/heads/oadtr.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import (Decoder, DecoderLayer, FullAttention, ProbAttention, AttentionLayer,
                    TransformerModel, FixedPositionalEncoding, LearnedPositionalEncoding)
from ..builder import HEADS


@HEADS.register()
class OadTRHead(nn.Module):
    def __init__(self,
                 clip_seg_num,
                 num_classes,
                 pred_clip_seg_num=8,
                 sample_rate=1,
                 patch_dim=1,
                 in_channels=2048,
                 embedding_dim=1024,
                 num_heads=8,
                 num_layers=3,
                 hidden_dim=1024,
                 decoder_embedding_dim=1024,
                 decoder_attn_dropout_rate=0.1,
                 decoder_num_heads=4,
                 decoder_layers=5,
                 decoder_embedding_dim_out=1024,
                 factor=1,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.1,
                 conv_patch_representation=False,
                 positional_encoding_type="learned"):
        super(OadTRHead, self).__init__()

        assert embedding_dim % num_heads == 0
        assert clip_seg_num % patch_dim == 0
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation
        self.clip_seg_num = clip_seg_num
        self.seq_length = self.clip_seg_num * 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )

        self.pe_dropout = nn.Dropout(p=self.dropout_rate)

        self.encoder = TransformerModel(
            embedding_dim,
            num_layers,
            num_heads,
            hidden_dim,
            self.dropout_rate,
            self.attn_dropout_rate,
        )
        self.pre_head_ln = nn.LayerNorm(embedding_dim)

        self.mlp_head = nn.Linear(embedding_dim + decoder_embedding_dim, num_classes)

        if self.conv_patch_representation:
            # self.conv_x = nn.Conv2d(
            #     self.in_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.in_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = nn.Conv1d(
                self.in_channels,
                self.embedding_dim,
                kernel_size=1)

        self.to_cls_token = nn.Identity()

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(FullAttention(True, factor, attention_dropout=decoder_attn_dropout_rate),  # True
                                   decoder_embedding_dim, decoder_num_heads),  # ProbAttention  FullAttention
                    AttentionLayer(FullAttention(False, factor, attention_dropout=decoder_attn_dropout_rate),  # False
                                   decoder_embedding_dim, decoder_num_heads),
                    decoder_embedding_dim,
                    decoder_embedding_dim_out,
                    dropout=decoder_attn_dropout_rate,
                    activation='gelu',
                )
                for l in range(decoder_layers)
            ],
            norm_layer=torch.nn.LayerNorm(decoder_embedding_dim)
        )
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, pred_clip_seg_num, decoder_embedding_dim))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                pred_clip_seg_num, self.embedding_dim, pred_clip_seg_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        self.classifier = nn.Linear(decoder_embedding_dim, num_classes)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(self, x, masks):
        # x.shape     [N C T]
        # masks.shape [N C T]

        # reducing channels
        x = self.conv_x(x) * masks[:, 0:1, ::self.sample_rate]
        # reshape 
        x = x.transpose(-1, -2).contiguous()
        # x.shape     [N T C]
        # masks.shape [N T C]
        cls_tokens = self.cls_token.expand(x.shape[0], x.shape[1], -1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)   # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [N, T, C]

        # decoder prediction layer
        # [N, pred_clip_seg_num, C]
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)

        pred_frames_feature = self.decoder(decoder_cls_token, x)   # [N, pred_clip_seg_num, C]
        pred_frames_feature = self.after_dropout(pred_frames_feature)
        pred_frames_for_token = pred_frames_feature.mean(dim=1).unsqueeze(1).expand(-1, self.clip_seg_num, -1)
        # [N, pred_clip_seg_num, C]
        pred_frames_score = self.classifier(pred_frames_feature)

        # classification layer
        # [N, T, C]
        x = torch.cat((self.to_cls_token(x[:, self.clip_seg_num:]), pred_frames_for_token), dim=-1)
        frames_score = self.mlp_head(x)

        # x: current chunck action
        # dec_cls_out: frame level action
        # [N, C, pred_clip_seg_num]
        pred_frames_score = pred_frames_score.transpose(-1, -2)
        # [N, C, T]
        frames_score = frames_score.transpose(-1, -2) * masks[:, 0:1, ::self.sample_rate]
        # [stage_num, N, C, T]
        frames_score = frames_score.unsqueeze(0)
        frames_score = F.interpolate(
            input=frames_score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        outputs = frames_score, pred_frames_score
        return outputs

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)