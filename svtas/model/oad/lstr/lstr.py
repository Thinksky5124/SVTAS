'''
Author       : Thyssen Wen
Date         : 2022-11-03 15:59:05
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-30 16:04:16
Description  : ref:https://github.com/amazon-science/long-short-term-transformer/blob/main/src/rekognition_online_action_detection/models/lstr.py
FilePath     : /SVTAS/svtas/model/heads/oad/lstr/lstr.py
'''
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import MultiheadAttention
from .transformer import Transformer
from .transformer import TransformerEncoder, TransformerEncoderLayer
from .transformer import TransformerDecoder, TransformerDecoderLayer
from .multihead_attention import layer_norm, generate_square_subsequent_mask

from svtas.utils import AbstractBuildFactory


class BaseFeatureHead(nn.Module):

    def __init__(self,
                 modality='twostream',
                 visual_size=1024,
                 motion_size=1024,
                 linear_enabled=True,
                 linear_out_features=1024
                 ):
        super(BaseFeatureHead, self).__init__()

        if modality in ['visual', 'motion', 'twostream']:
            self.with_visual = 'motion' not in modality
            self.with_motion = 'visual' not in modality
        else:
            raise RuntimeError('Unknown modality of {}'.format(modality))

        if self.with_visual and self.with_motion:
            fusion_size = visual_size + motion_size
        elif self.with_visual:
            fusion_size = visual_size
        elif self.with_motion:
            fusion_size = motion_size

        self.d_model = fusion_size

        if linear_enabled:
            if linear_out_features != -1:
                self.d_model = linear_out_features
            self.input_linear = nn.Sequential(
                nn.Linear(fusion_size, self.d_model),
                nn.ReLU(inplace=True),
            )
        else:
            self.input_linear = nn.Identity()

    def forward(self, visual_input, motion_input):
        if self.with_visual and self.with_motion:
            fusion_input = torch.cat((visual_input, motion_input), dim=-1)
        elif self.with_visual:
            fusion_input = visual_input
        elif self.with_motion:
            fusion_input = motion_input
        fusion_input = self.input_linear(fusion_input)
        return fusion_input


@AbstractBuildFactory.register('model')
class LSTR(nn.Module):

    def __init__(self,
                 modality='twostream',
                 visual_size=1024,
                 motion_size=1024,
                 linear_enabled=True,
                 linear_out_features=1024,
                 long_memory_num_samples=512,
                 work_memory_num_samples=32,
                 num_heads=16,
                 dim_feedforward=1024,
                 dropout=0.2,
                 activation='relu',
                 num_classes=11,
                 enc_module=[
                            [16, 1, True], [32, 2, True]
                            ],
                 dec_module=[-1, 2, True],
                 sample_rate=1,
                 max_memory_length=2048):
        super(LSTR, self).__init__()
        self.sample_rate = sample_rate
        self.visual_size = visual_size
        self.motion_size = motion_size
        self.max_memory_length = max_memory_length

        # Build long feature heads
        self.long_memory_num_samples = long_memory_num_samples
        self.long_enabled = self.long_memory_num_samples > 0
        if self.long_enabled:
            self.feature_head_long = BaseFeatureHead(modality=modality,
                 visual_size=visual_size,
                 motion_size=motion_size,
                 linear_enabled=linear_enabled,
                 linear_out_features=linear_out_features)

        # Build work feature head
            
        self.work_memory_num_samples = work_memory_num_samples
        self.work_enabled = self.work_memory_num_samples > 0
        if self.work_enabled:
            self.feature_head_work = BaseFeatureHead(modality=modality,
                 visual_size=visual_size,
                 motion_size=motion_size,
                 linear_enabled=linear_enabled,
                 linear_out_features=linear_out_features)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.activation = activation
        self.num_classes = num_classes

        # Build position encoding
        self.pos_encoding = FixedPositionalEncoding(self.d_model, self.dropout)

        # Build LSTR encoder
        if self.long_enabled:
            self.enc_queries = nn.ModuleList()
            self.enc_modules = nn.ModuleList()
            for param in enc_module:
                if param[0] != -1:
                    self.enc_queries.append(nn.Embedding(param[0], self.d_model))
                    enc_layer = TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerDecoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
                else:
                    self.enc_queries.append(None)
                    enc_layer = TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout, self.activation)
                    self.enc_modules.append(TransformerEncoder(
                        enc_layer, param[1], layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('enc_queries', None)
            self.register_parameter('enc_modules', None)

        # Build LSTR decoder
        if self.long_enabled:
            param = dec_module
            dec_layer = TransformerDecoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerDecoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))
        else:
            param = dec_module
            dec_layer = TransformerEncoderLayer(
                self.d_model, self.num_heads, self.dim_feedforward,
                self.dropout, self.activation)
            self.dec_modules = TransformerEncoder(
                dec_layer, param[1], layer_norm(self.d_model, param[2]))

        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

        ############################
        # Cache for stream inference
        ############################
        self.long_memories_cache = None
        self.compressed_long_memories_cache = None
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data, a=math.sqrt(5))
        elif isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose1d):
            nn.init.normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.xavier_normal_(m.weight.data)
            if m.bias is not None:
                nn.init.normal_(m.bias.data)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, mean=1, std=0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def init_weights(self, init_cfg: dict = {}):
        self.apply(self.weights_init)

    def _clear_memory_buffer(self):
        pass

    def stream_train(self, feature, masks):
        visual_inputs = feature[:, :self.visual_size, :].transpose(1, 2)
        motion_inputs = feature[:, self.motion_size:, :].transpose(1, 2)
        masks = masks[:, 0:1, ::self.sample_rate]

        # Get memory key padding mask
        if self.long_memory_num_samples > 0:
            memory_key_padding_mask = masks[:, 0, :self.long_memory_num_samples]
        else:
            memory_key_padding_mask = None

        if self.long_enabled:
            # Compute long memories
            long_memories = self.pos_encoding(self.feature_head_long(
                visual_inputs[:, :self.long_memory_num_samples],
                motion_inputs[:, :self.long_memory_num_samples],
            ).transpose(0, 1))

            if len(self.enc_modules) > 0:
                enc_queries = [
                    enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                    if enc_query is not None else None
                    for enc_query in self.enc_queries
                ]

                # Encode long memories
                if enc_queries[0] is not None:
                    long_memories = self.enc_modules[0](enc_queries[0], long_memories,
                                                        memory_key_padding_mask=memory_key_padding_mask)
                else:
                    long_memories = self.enc_modules[0](long_memories)
                for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                    if enc_query is not None:
                        long_memories = enc_module(enc_query, long_memories)
                    else:
                        long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                visual_inputs[:, self.long_memory_num_samples:],
                motion_inputs[:, self.long_memory_num_samples:],
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        # [T N C]
        score = self.classifier(output)
        # [T N C] -> [stage_num， N, C, T]
        score = torch.permute(score, dims=[1, 2, 0]).unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        return score
    
    def stream_inference(self, feature, masks=None):
        assert self.long_enabled, 'Long-term memory cannot be empty for stream inference'
        assert len(self.enc_modules) > 0, 'LSTR encoder cannot be disabled for stream inference'
        masks = masks[:, 0:1, ::self.sample_rate]
        long_visual_inputs = feature[:, :self.visual_size, :self.long_memory_num_samples].transpose(1, 2)
        long_motion_inputs = feature[:, self.motion_size:, :self.long_memory_num_samples].transpose(1, 2)
        work_visual_inputs = feature[:, :self.visual_size, self.long_memory_num_samples:].transpose(1, 2)
        work_motion_inputs = feature[:, self.motion_size:, self.long_memory_num_samples:].transpose(1, 2)
        # Get memory key padding mask
        if self.long_memory_num_samples > 0:
            memory_key_padding_mask = masks[:, 0, :self.long_memory_num_samples]
        else:
            memory_key_padding_mask = None

        if (long_visual_inputs is not None) and (long_motion_inputs is not None):
            # Compute long memories [T N C]
            long_memories = self.feature_head_long(
                long_visual_inputs,
                long_motion_inputs,
            ).transpose(0, 1)

            if self.long_memories_cache is None:
                self.long_memories_cache = long_memories.detach().clone()
            else:
                self.long_memories_cache = torch.cat((
                    self.long_memories_cache[self.work_memory_num_samples:], long_memories.detach().clone()
                ))[max(0, self.long_memories_cache.shape[0] - self.max_memory_length):]
                

            long_memories = self.long_memories_cache
            pos = self.pos_encoding.pe[:self.long_memory_num_samples, :]

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            long_memories = self.enc_modules[0].stream_inference(enc_queries[0], long_memories, pos,
                                                                 memory_key_padding_mask=memory_key_padding_mask)
            self.compressed_long_memories_cache  = long_memories.detach().clone()
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)
        else:
            long_memories = self.compressed_long_memories_cache

            enc_queries = [
                enc_query.weight.unsqueeze(1).repeat(1, long_memories.shape[1], 1)
                if enc_query is not None else None
                for enc_query in self.enc_queries
            ]

            # Encode long memories
            for enc_query, enc_module in zip(enc_queries[1:], self.enc_modules[1:]):
                if enc_query is not None:
                    long_memories = enc_module(enc_query, long_memories)
                else:
                    long_memories = enc_module(long_memories)

        # Concatenate memories
        if self.long_enabled:
            memory = long_memories

        if self.work_enabled:
            # Compute work memories
            work_memories = self.pos_encoding(self.feature_head_work(
                work_visual_inputs,
                work_motion_inputs,
            ).transpose(0, 1), padding=self.long_memory_num_samples)

            # Build mask
            mask = generate_square_subsequent_mask(
                work_memories.shape[0])
            mask = mask.to(work_memories.device)

            # Compute output
            if self.long_enabled:
                output = self.dec_modules(
                    work_memories,
                    memory=memory,
                    tgt_mask=mask,
                )
            else:
                output = self.dec_modules(
                    work_memories,
                    src_mask=mask,
                )

        # Compute classification score
        # [T N C]
        score = self.classifier(output)
        # [T N C] -> [stage_num， N, C, T]
        score = torch.permute(score, dims=[1, 2, 0]).unsqueeze(0)
        score = F.interpolate(
            input=score,
            scale_factor=[1, self.sample_rate],
            mode="nearest")
        return score

    def forward(self, feature, masks):
        if self.training:
            return self.stream_train(feature, masks)
        else:
            return self.stream_inference(feature, masks)