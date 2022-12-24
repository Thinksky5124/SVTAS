'''
Author       : Thyssen Wen
Date         : 2022-12-23 17:50:38
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-24 22:20:11
Description  : file content
FilePath     : /SVTAS/svtas/utils/cam/model_wapper.py
'''
import torch

class ModelForwardWrapper(torch.nn.Module):
    def __init__(self,
                 model,
                 data_key,
                 sample_rate): 
        super(ModelForwardWrapper, self).__init__()
        self.model = model
        self.sample_rate = sample_rate
        self.data_key = data_key
        
    def forward(self, inpute_tensor):
        masks = torch.full([inpute_tensor.shape[0], inpute_tensor.shape[1] * self.sample_rate], 1.0).to(inpute_tensor.device)
        return [self.model({self.data_key:inpute_tensor, "masks":masks})["output"]]