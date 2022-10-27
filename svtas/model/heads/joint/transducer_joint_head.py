'''
Author       : Thyssen Wen
Date         : 2022-05-21 13:47:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-27 21:02:50
Description  : Transudcer JointNet ref:https://github.com/upskyy/Transformer-Transducer/blob/main/transformer_transducer/model.py
FilePath     : /ETESVS/model/heads/transducer_joint_head.py
'''
from torch import Tensor
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import HEADS

@HEADS.register()
class TransudcerJointNet(nn.Module):
    """
    Combine the audio encoder and label encoders.
    Convert them into log probability values for each word.
    Args:
        num_vocabs (int): the number of vocabulary
        output_size (int): the number of features combined output of audio and label encoders (default : 1024)
        inner_size (int): the number of inner features (default : 512)
    Inputs: audio_encoder, label_encoder
        - **audio_encoder**: Audio encoder output
        - **label_encoder**: Label encoder output
    Returns: output
        - **output**: Tensor expressing the log probability values of each word
    """
    def __init__(
            self,
            num_classes: int,
            in_channels: int = 1024,
            hidden_channels: int = 512,
    ) -> None:
        super(TransudcerJointNet, self).__init__()
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(hidden_channels, num_classes)
    
    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass

    def forward(
            self,
            audio_encoder: Tensor,
            label_encoder: Tensor,
    ) -> Tensor:
        if audio_encoder.dim() == 3 and label_encoder.dim() == 3:  # Train
            seq_lens = audio_encoder.size(1)
            target_lens = label_encoder.size(1)

            audio_encoder = audio_encoder.unsqueeze(2)
            label_encoder = label_encoder.unsqueeze(1)

            audio_encoder = audio_encoder.repeat(1, 1, target_lens, 1)
            label_encoder = label_encoder.repeat(1, seq_lens, 1, 1)

        output = torch.cat((audio_encoder, label_encoder), dim=-1)

        output = self.fc1(output)
        output = self.tanh(output)
        output = self.fc2(output)

        output = F.log_softmax(output, dim=-1)

        return output