'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:02:01
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 10:00:46
Description  : Transducer AudioEncoder ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/audio_encoder.py
FilePath     : /ETESVS/model/backbones/audio/transudcer_audio_encoder.py
'''
from typing import Tuple
from torch import Tensor
from ..utils.transducer import get_attn_pad_mask, PositionalEncoding, EncoderLayer
import torch
import torch.nn as nn

from utils.logger import get_logger
from mmcv.runner import load_checkpoint
from ...builder import BACKBONES

@BACKBONES.register()
class TransducerAudioEncoder(nn.Module):
    """
    Converts the audio signal to higher feature values
    Args:
        device (torch.device): flag indication whether cpu or cuda
        input_size (int): dimension of input vector (default : 80)
        model_dim (int): the number of features in the audio encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of audio encoder layers (default: 18)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of audio encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)
    Inputs: inputs, inputs_lens
        - **inputs**: Parsed audio of batch size number
        - **inputs_lens**: Tensor of sequence lengths
    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """
    def __init__(
            self,
            device: torch.device,
            input_size: int = 80,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 18,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
    ) -> None:
        super(TransducerAudioEncoder, self).__init__()
        self.device = device
        self.input_size = input_size
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_fc = nn.Linear(input_size, model_dim)
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("SVTAS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for audio encoder.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to audio encoder. Typically inputs will be a padded
                `FloatTensor` of size ``(batch, dimension, seq_length)``.
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        inputs = inputs.transpose(1, 2)
        seq_len = inputs.size(1)

        self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, seq_len)

        inputs = self.input_fc(inputs) + self.positional_encoding(seq_len)
        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs