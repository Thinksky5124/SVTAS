'''
Author       : Thyssen Wen
Date         : 2022-05-21 11:03:04
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-21 11:05:30
Description  : Transducer TextEncoder ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/label_encoder.py
FilePath     : /ETESVS/model/backbones/text/transducer_text_encoder.py
'''
from typing import Tuple
from torch import Tensor
from ..utils.transducer import get_attn_pad_mask, PositionalEncoding, EncoderLayer
import torch
import torch.nn as nn
import numpy as np

from utils.logger import get_logger
from mmcv.runner import load_checkpoint
from ...builder import BACKBONES

@BACKBONES.register()
class TransducerTextEncoder(nn.Module):
    """
    Converts the label to higher feature values
    Args:
        device (torch.device): flag indication whether cpu or cuda
        num_vocabs (int): the number of vocabulary
        model_dim (int): the number of features in the label encoder (default : 512)
        ff_dim (int): the number of features in the feed forward layers (default : 2048)
        num_layers (int): the number of label encoder layers (default: 2)
        num_heads (int): the number of heads in the multi-head attention (default: 8)
        dropout (float): dropout probability of label encoder (default: 0.1)
        max_len (int): Maximum length to use for positional encoding (default : 5000)
        pad_id (int): index of padding (default: 0)
        sos_id (int): index of the start of sentence (default: 1)
        eos_id (int): index of the end of sentence (default: 2)
    Inputs: inputs, inputs_lens
        - **inputs**: Ground truth of batch size number
        - **inputs_lens**: Tensor of target lengths
    Returns: outputs
        - **outputs**: Tensor containing higher feature values
    """
    def __init__(
            self,
            device: torch.device,
            num_vocabs: int,
            model_dim: int = 512,
            ff_dim: int = 2048,
            num_layers: int = 2,
            num_heads: int = 8,
            dropout: float = 0.1,
            max_len: int = 5000,
            pad_id: int = 0,
            sos_id: int = 1,
            eos_id: int = 2,
    ) -> None:
        super(TransducerTextEncoder, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(num_vocabs, model_dim)
        self.scale = np.sqrt(model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_len)
        self.input_dropout = nn.Dropout(p=dropout)
        self.pad_id = pad_id
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, ff_dim, num_heads, dropout) for _ in range(num_layers)]
        )
    
    def _clear_memory_buffer(self):
        pass
    
    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)

    def forward(
            self,
            inputs: Tensor,
            inputs_lens: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward propagate a `inputs` for label encoder.
        Args:
            inputs (torch.LongTensor): A input sequence passed to label encoder. Typically inputs will be a padded
                `LongTensor` of size ``(batch, target_length)``
            inputs_lens (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            **outputs** (Tensor): ``(batch, seq_length, dimension)``
        """
        self_attn_mask = None
        batch = inputs.size(0)

        if len(inputs.size()) == 1:  # validate, evaluation
            inputs = inputs.unsqueeze(1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs).to(self.device) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

        else:  # train
            inputs = inputs[inputs != self.eos_id].view(batch, -1)
            target_lens = inputs.size(1)

            embedding_output = self.embedding(inputs).to(self.device) * self.scale
            positional_encoding_output = self.positional_encoding(target_lens)
            inputs = embedding_output + positional_encoding_output

            self_attn_mask = get_attn_pad_mask(inputs, inputs_lens, target_lens)

        outputs = self.input_dropout(inputs)

        for encoder_layer in self.encoder_layers:
            outputs, _ = encoder_layer(outputs, self_attn_mask)

        return outputs