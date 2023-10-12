'''
Author       : Thyssen Wen
Date         : 2022-05-17 15:18:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-03 20:13:03
Description  : PositionEncoding utils ref:https://github.com/wangxiang1230/OadTR/blob/main/transformer_models/PositionalEncoding.py
FilePath     : /SVTAS/svtas/model/heads/utils/oadtr/position_encoding.py
'''
import torch
import torch.nn as nn


class FixedPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, dropout=0.1, max_length=5000):
        super(FixedPositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_length, embedding_dim)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, padding=0):
        x = x + self.pe[padding: padding + x.shape[0], :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1)),
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, :self.seq_length]

        position_embeddings = self.pe(position_ids)
        return x + position_embeddings