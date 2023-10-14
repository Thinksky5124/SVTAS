'''
Author       : Thyssen Wen
Date         : 2023-10-14 15:29:32
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-14 15:33:06
Description  : file content
FilePath     : /SVTAS/svtas/model/vae/tas_vae.py
'''
from ..architectures.general import VariationalAutoEncoder
from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('model')
class TemporalActionSegmentationVariationalAutoEncoder(VariationalAutoEncoder):
    def __init__(self,
                 encoder,
                 decoder,
                 weight_init_cfg=dict(
                    encoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    decoder = dict(
                        child_model=False,
                        revise_keys=[(r'backbone.', r'')]),
                    )):
        super().__init__(encoder, decoder, weight_init_cfg)
    
    def forward(self, input_data):
        return super().forward(input_data)