'''
Author       : Thyssen Wen
Date         : 2022-05-21 10:47:47
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-21 10:47:50
Description  : Transducer mask function ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/mask.py
FilePath     : /ETESVS/model/backbones/utils/transducer/mask.py
'''
from torch import Tensor


def _get_pad_mask(inputs: Tensor, inputs_lens: Tensor):
    assert len(inputs.size()) == 3

    batch = inputs.size(0)

    pad_attn_mask = inputs.new_zeros(inputs.size()[: -1])

    for idx in range(batch):
        pad_attn_mask[idx, inputs_lens[idx]:] = 1

    return pad_attn_mask.bool()


def get_attn_pad_mask(inputs: Tensor, inputs_lens: Tensor, expand_lens):
    pad_attn_mask = _get_pad_mask(inputs, inputs_lens)
    pad_attn_mask = pad_attn_mask.unsqueeze(1).repeat(1, expand_lens, 1)  # (batch, dec_T, enc_T)

    return pad_attn_mask