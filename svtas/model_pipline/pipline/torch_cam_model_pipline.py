'''
Author       : Thyssen Wen
Date         : 2023-10-08 15:29:59
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 16:14:01
Description  : file content
FilePath     : /SVTAS/svtas/model_pipline/pipline/torch_cam_model_pipline.py
'''
import math
import torch
from typing import Dict
from .torch_model_pipline import TorchModelPipline
from svtas.utils.cam import get_model_target_class
from pytorch_grad_cam.ablation_layer import AblationLayerVit
from svtas.utils.cam import ModelForwardWrapper, get_match_fn_class
from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

def reshape_transform(transform_form):
# # class activation transform [N C T]
    def reshape_transform_NCT(tensor):
        # [N C T] -> [N C T 1]
        result = tensor.unsqueeze(-1)
        return result

    # feature activation transform [N P C]
    def reshape_transform_NPC(tensor):
        # for padding cls_token
        # result = tensor[:, 1:, :].reshape(tensor.size(0), int(math.sqrt(tensor.size(1))),
        #                                   int(math.sqrt(tensor.size(1))), tensor.size(2))
        # for image
        result = tensor.reshape(tensor.size(0), int(math.sqrt(tensor.size(1))),
                                          int(math.sqrt(tensor.size(1))), tensor.size(2))

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # feature activation transform [N C T H W]
    def reshape_transform_NCTHW(tensor, height=7, width=7):
        result = torch.permute(tensor, [0, 2, 3, 4, 1])
        result = torch.reshape(result, [-1, height, width, result.shape[-1]])

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result
        
    if transform_form == "NCT":
        return reshape_transform_NCT
    elif transform_form == "NPC":
        return reshape_transform_NPC
    elif transform_form == "NCTHW":
        return reshape_transform_NCTHW
    else:
        print("Not support form!")
        raise NotImplementedError
    
class TorchCAMModelPipline(TorchModelPipline):
    def __init__(self,
                 model,
                 post_processing,
                 cam_method,
                 eigen_smooth,
                 aug_smooth,
                 method,
                 batch_size,
                 sample_rate,
                 ignore_index = -100,
                 layer_name = [],
                 data_key = "imgs",
                 return_targets_name = dict(
                     CategorySegmentationTarget = dict(category=None)
                 ),
                 reshape_transform = "NPC",
                 label_path = "./data/gtea/mapping.txt",
                 match_fn = "rgb_stream_match_fn",
                 device=None,
                 criterion=None,
                 optimizer=None,
                 lr_scheduler=None,
                 pretrained: str = None,
                 amp: Dict = None,
                 grad_clip: Dict = None,
                 grad_accumulate: Dict = None) -> None:
        super().__init__(model, post_processing, device, criterion, optimizer,
                         lr_scheduler, pretrained, amp, grad_clip, grad_accumulate)
        self.model = ModelForwardWrapper(model=self.model, data_key=data_key, sample_rate=sample_rate)
        self.methods = \
            {"gradcam": GradCAM,
            "scorecam": ScoreCAM,
            "gradcam++": GradCAMPlusPlus,
            "ablationcam": AblationCAM,
            "xgradcam": XGradCAM,
            "eigencam": EigenCAM,
            "eigengradcam": EigenGradCAM,
            "layercam": LayerCAM,
            "fullgrad": FullGrad}

        if method not in list(self.methods.keys()):
            raise Exception(f"method should be one of {list(self.methods.keys())}")
        
        self.cam_method = cam_method
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
        self.method = method
        self.label_path = label_path
        self.data_key = data_key
        self.return_targets_name = return_targets_name
        self.match_fn = get_match_fn_class(match_fn)
        self.layer_name = layer_name
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.reshape_transform = reshape_transform

        self.target_layers = []
        # batch videos sampler
        for layer in self.model_pipline.model.named_modules():
            if layer[0] in set(self.layer_name):
                self.target_layers.append(layer[1])

        if self.cam_method == "ablationcam":
            self.cam = self.methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.reshape_transform),
                                    ablation_layer=AblationLayerVit())
        else:
            self.cam = self.methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.reshape_transform))
        self.cam.batch_size = self.batch_size
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        if self.return_targets_name is None:
            self.targets = None
        else:
            self.targets = []
            for k, cfg in self.return_targets_name.items():
                self.targets.append(get_model_target_class(target_name=k, cfg=cfg))

    @torch.no_grad()
    def forward(self, data_dict):
        input_tensor = input_data[self.data_key]
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                if torch.cuda.is_available():
                    input_data[key] = value.to(self.device)
                else:
                    input_data[key] = value
        if not self.grad_accumulate:
            input_data['precise_sliding_num'] = torch.ones_like(input_data['precise_sliding_num'])

        outputs = self.model(input_tensor)
        grayscale_cam = self.cam(input_tensor=input_tensor,
                            targets=self.targets,
                            eigen_smooth=self.eigen_smooth,
                            aug_smooth=self.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        cam_images = self.match_fn(data_dict, grayscale_cam)
        outputs['cam_images'] = cam_images
        return outputs, input_data