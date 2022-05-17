'''
Author       : Thyssen Wen
Date         : 2022-05-17 14:57:48
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-17 16:54:28
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
                 sample_rate=1,
                 patch_dim=1,
                 embedding_dim=1024,
                 num_heads=8,
                 num_layers=3,
                 hidden_dim=1024,
                 decoder_embedding_dim=1024,
                 query_num=8,
                 decoder_attn_dropout_rate=0.1,
                 decoder_num_heads=4,
                 decoder_layers=5,
                 decoder_embedding_dim_out=1024,
                 factor=1,
                 dropout_rate=0.1,
                 attn_dropout_rate=0.1,
                 conv_patch_representation=False,
                 positional_encoding_type="learned",
                 num_channels=3072):
        super(OadTRHead, self).__init__()

        assert embedding_dim % num_heads == 0
        assert clip_seg_num % patch_dim == 0
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        # num_channels = clip_seg_num
        self.num_channels = num_channels
        self.dropout_rate = dropout_rate
        self.attn_dropout_rate = attn_dropout_rate
        self.conv_patch_representation = conv_patch_representation

        # self.num_patches = int((clip_seg_num // patch_dim) ** 2)
        self.num_patches = int(clip_seg_num // patch_dim)
        self.seq_length = self.num_patches + 1
        self.flatten_dim = patch_dim * patch_dim * num_channels
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))

        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        if positional_encoding_type == "learned":
            self.position_encoding = LearnedPositionalEncoding(
                self.seq_length, self.embedding_dim, self.seq_length
            )
        elif positional_encoding_type == "fixed":
            self.position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position encoding :', positional_encoding_type)

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
            #     self.num_channels,
            #     self.embedding_dim,
            #     kernel_size=(self.patch_dim, self.patch_dim),
            #     stride=(self.patch_dim, self.patch_dim),
            #     padding=self._get_padding(
            #         'VALID', (self.patch_dim, self.patch_dim),
            #     ),
            # )
            self.conv_x = nn.Conv1d(
                self.num_channels,
                self.embedding_dim,
                kernel_size=self.patch_dim,
                stride=self.patch_dim,
                padding=self._get_padding(
                    'VALID',  (self.patch_dim),
                ),
            )
        else:
            self.conv_x = None

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
        self.decoder_cls_token = nn.Parameter(torch.zeros(1, query_num, decoder_embedding_dim))
        if positional_encoding_type == "learned":
            self.decoder_position_encoding = LearnedPositionalEncoding(
                query_num, self.embedding_dim, query_num
            )
        elif positional_encoding_type == "fixed":
            self.decoder_position_encoding = FixedPositionalEncoding(
                self.embedding_dim,
            )
        print('position decoding :', positional_encoding_type)
        self.classifier = nn.Linear(decoder_embedding_dim, num_classes)
        self.after_dropout = nn.Dropout(p=self.dropout_rate)
        # self.merge_fc = nn.Linear(d_model, 1)
        # self.merge_sigmoid = nn.Sigmoid()

    def forward(self, x, masks):
        # x.shape     [N C T]
        # masks.shape [N C T]
        x = self.linear_encoding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        # x = torch.cat((cls_tokens, x), dim=1)
        x = torch.cat((x, cls_tokens), dim=1)
        x = self.position_encoding(x)
        x = self.pe_dropout(x)   # not delete

        # apply transformer
        x = self.encoder(x)
        x = self.pre_head_ln(x)  # [128, 33, 1024]
        # x = self.after_dropout(x)  # add
        # decoder
        decoder_cls_token = self.decoder_cls_token.expand(x.shape[0], -1, -1)
        # decoder_cls_token = self.after_dropout(decoder_cls_token)  # add
        # decoder_cls_token = self.decoder_position_encoding(decoder_cls_token)  # [128, 8, 1024]
        dec = self.decoder(decoder_cls_token, x)   # [128, 8, 1024]
        dec = self.after_dropout(dec)  # add
        # merge_atte = self.merge_sigmoid(self.merge_fc(dec))  # [128, 8, 1]
        # dec_for_token = (merge_atte*dec).sum(dim=1)  # [128, 1024]
        # dec_for_token = (merge_atte*dec).sum(dim=1)/(merge_atte.sum(dim=-2) + 0.0001)
        dec_for_token = dec.mean(dim=1)
        # dec_for_token = dec.max(dim=1)[0]
        frame_score = self.classifier(dec)
        # x = self.to_cls_token(x[:, 0])
        x = torch.cat((self.to_cls_token(x[:, -1]), dec_for_token), dim=1)
        chunck_action = self.mlp_head(x)
        # x = F.log_softmax(x, dim=-1)

        # x: current chunck action
        # dec_cls_out: frame level action
        frame_score = F.interpolate(
            input=frame_score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        outputs = frame_score, chunck_action
        return outputs

    def _get_padding(self, padding_type, kernel_size):
        assert padding_type in ['SAME', 'VALID']
        if padding_type == 'SAME':
            _list = [(k - 1) // 2 for k in kernel_size]
            return tuple(_list)
        return tuple(0 for _ in kernel_size)