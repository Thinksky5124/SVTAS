'''
Author       : Thyssen Wen
Date         : 2022-06-13 14:42:47
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-17 10:28:03
Description  : ConFormer Head for Action Segmentation ref:https://github.com/sooftware/conformer/blob/main/conformer/model.py
FilePath     : /SVTAS/svtas/model/asr/conformer/conformer.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .encoder import ConformerEncoder, ConformerDecoder
from .modules import ConFormerLinear

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class Conformer(nn.Module):
    """
    Conformer: Convolution-augmented Transformer for Speech Recognition
    The paper used a one-lstm Transducer decoder, currently still only implemented
    the conformer encoder shown in the paper.
    Args:
        num_classes (int): Number of classification classes
        input_dim (int, optional): Dimension of input vector
        encoder_dim (int, optional): Dimension of conformer encoder
        num_encoder_layers (int, optional): Number of conformer blocks
        num_attention_heads (int, optional): Number of attention heads
        feed_forward_expansion_factor (int, optional): Expansion factor of feed forward module
        conv_expansion_factor (int, optional): Expansion factor of conformer convolution module
        feed_forward_dropout_p (float, optional): Probability of feed forward module dropout
        attention_dropout_p (float, optional): Probability of attention module dropout
        conv_dropout_p (float, optional): Probability of conformer convolution module dropout
        conv_kernel_size (int or tuple, optional): Size of the convolving kernel
        half_step_residual (bool): Flag indication whether to use half step residual or not
    Inputs: inputs
        - **inputs** (batch, time, dim): Tensor containing input vector
        - **input_lengths** (batch): list of sequence input lengths
    Returns: outputs, output_lengths
        - **outputs** (batch, out_channels, time): Tensor produces by conformer.
        - **output_lengths** (batch): list of sequence output lengths
    """
    def __init__(
            self,
            num_classes: int,
            sample_rate: int = 1,
            out_feature: bool = False,
            input_dim: int = 80,
            encoder_dim: int = 512,
            num_encoder_layers: int = 17,
            num_stages: int = 3,
            num_attention_heads: int = 8,
            feed_forward_expansion_factor: int = 4,
            conv_expansion_factor: int = 2,
            input_dropout_p: float = 0.1,
            feed_forward_dropout_p: float = 0.1,
            attention_dropout_p: float = 0.1,
            conv_dropout_p: float = 0.1,
            conv_kernel_size: int = 31,
            half_step_residual: bool = True,
            need_subsampling: bool = True,
    ) -> None:
        super(Conformer, self).__init__()
        self.sample_rate = sample_rate
        self.out_feature = out_feature
        self.num_classes = num_classes
        self.embedding = nn.Conv1d(in_channels=input_dim, out_channels=encoder_dim, kernel_size=1)
        
        self.encoder = ConformerEncoder(
            input_dim=encoder_dim,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            input_dropout_p=input_dropout_p,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
            need_subsampling = need_subsampling,
        )
        self.decoders = nn.ModuleList([ConformerDecoder(
            input_dim=num_classes,
            encoder_dim=encoder_dim,
            num_layers=num_encoder_layers,
            num_attention_heads=num_attention_heads,
            feed_forward_expansion_factor=feed_forward_expansion_factor,
            conv_expansion_factor=conv_expansion_factor,
            feed_forward_dropout_p=feed_forward_dropout_p,
            attention_dropout_p=attention_dropout_p,
            conv_dropout_p=conv_dropout_p,
            conv_kernel_size=conv_kernel_size,
            half_step_residual=half_step_residual,
        ) for _ in range(num_stages - 1)])
        self.cls = ConFormerLinear(encoder_dim, num_classes, bias=False)

    def init_weights(self, init_cfg: dict = {}):
        pass

    def _clear_memory_buffer(self):
        pass

    def count_parameters(self) -> int:
        """ Count parameters of encoder """
        return self.encoder.count_parameters()

    def update_dropout(self, dropout_p) -> None:
        """ Update dropout probability of model """
        self.encoder.update_dropout(dropout_p)

    def forward(self, inputs, masks):
        """
        Forward propagate a `inputs` and `targets` pair for training.
        Args:
            inputs (torch.FloatTensor): A input sequence passed to encoder. Typically for inputs this will be a padded
                `FloatTensor` of size ``(batch, seq_length, dimension)``.
            input_lengths (torch.LongTensor): The length of input tensor. ``(batch)``
        Returns:
            * predictions (torch.FloatTensor): Result of model predictions.
        """
        # inputs [N D T]
        # masks [N D T]
        inputs = self.embedding(inputs)
        inputes_T = torch.permute(inputs, dims=[0, 2, 1])
        input_lengths = torch.tensor([inputs.shape[-1] for _ in range(inputs.shape[0])]).to(inputs.device)
        encoder_outputs, encoder_output_lengths = self.encoder(inputes_T, input_lengths)
        out = self.cls(encoder_outputs)
        outputs = out.unsqueeze(0)
        for decoder in self.decoders:
            out = decoder(F.softmax(out, dim=2), encoder_outputs)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        outputs = torch.permute(outputs, dims=[0, 1, 3, 2])

        # pool shape
        outputs = F.interpolate(
            input=outputs,
            size=[self.num_classes, masks.shape[-1]],
            mode="nearest") * masks[:, 0:1, :]
            
        if self.out_feature is True:
            return inputs, outputs
        return outputs