'''
Author       : Thyssen Wen
Date         : 2022-10-31 19:02:43
LastEditors  : Thyssen Wen
LastEditTime : 2022-12-24 22:05:20
Description  : file content
FilePath     : /SVTAS/svtas/runner/visual_runner.py
'''
import os
import math
import torch

from ..utils.flow_vis import make_palette
from ..utils.cam import get_model_target_class
from pytorch_grad_cam.ablation_layer import AblationLayerVit

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
        
class VisualRunner():
    def __init__(self,
                 cam_method,
                 use_cuda,
                 eigen_smooth,
                 aug_smooth,
                 logger,
                 model,
                 visualize_cfg,
                 post_processing,
                 cam_imgs_out_path,
                 methods,
                 match_fn):
        self.model = model
        self.logger = logger
        self.visualize_cfg = visualize_cfg
        self.post_processing = post_processing
        self.cam_imgs_out_path = cam_imgs_out_path
        self.cam_method = cam_method
        self.use_cuda = use_cuda
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
        self.methods = methods
        self.match_fn = match_fn
    
    def epoch_init(self):
        self.target_layers = []
        # batch videos sampler
        for layer in self.model.named_modules():
            if layer[0] in set(self.visualize_cfg.layer_name):
                self.target_layers.append(layer[1])

        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        if self.cam_method == "ablationcam":
            self.cam = self.methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.visualize_cfg.reshape_transform),
                                    ablation_layer=AblationLayerVit())
        else:
            self.cam = self.methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.visualize_cfg.reshape_transform))
        self.cam.batch_size = self.visualize_cfg.batch_size
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        if self.visualize_cfg.return_targets_name is None:
            self.targets = None
        else:
            self.targets = []
            for k, cfg in self.visualize_cfg.return_targets_name.items():
                self.targets.append(get_model_target_class(target_name=k, cfg=cfg))

        # load mapping label
        file_ptr = open(self.visualize_cfg.label_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        for a in actions:
            actions_dict[int(a.split()[0])] = a.split()[1]
        self.palette = make_palette(len(actions_dict))
        self.actions_dict = actions_dict
    
        self.model.eval()
    
    def batch_end_step(self, sliding_num, vid_list, step):

        # get extract feature
        cam_imgs_list, labels_list, preds_list = self.post_processing.output()
        
        # save feature file
        current_vid_list = self.current_step_vid_list
        for cam_imgs, vid, labels, preds in zip(cam_imgs_list, current_vid_list, labels_list, preds_list):
            cam_imgs_save_path = os.path.join(self.cam_imgs_out_path, vid + ".mp4")
            stream_writer, v_len = cam_imgs["writer"], cam_imgs["len"]
            stream_writer.save(cam_imgs_save_path, v_len, labels, preds, self.actions_dict, self.palette)

        self.logger.info("Step: " + str(step) + ", finish ectracting video: "+ ",".join(current_vid_list))
        self.current_step_vid_list = vid_list
        
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores()

        self.current_step = step
    
    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                input_data[key] = value.cuda()

        # AblationCAM and ScoreCAM have batched implementations.
        # You can override the internal batch size for faster computation.
        input_tensor = input_data[self.visualize_cfg.data_key]
            
        with torch.no_grad():
            outputs = self.model(input_tensor)
            score = outputs[0]

        grayscale_cam = self.cam(input_tensor=input_tensor,
                            targets=self.targets,
                            eigen_smooth=self.eigen_smooth,
                            aug_smooth=self.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        cam_images = self.match_fn(data_dict, grayscale_cam)
        return score, cam_images
    
    def run_one_clip(self, data_dict):
        vid_list = data_dict['vid_list']
        idx = data_dict['current_sliding_cnt']
        labels = data_dict['labels']
        # train segment
        score, cam_images = self._model_forward(data_dict)
            
        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores()
                self.current_step_vid_list = vid_list
            self.post_processing.update(cam_images, labels, score, idx)

    def run_one_iter(self, data):
        # videos sliding stream train
        for sliding_seg in data:
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step
            if self.current_step != step or (len(vid_list) <= 0 and step == 1):
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step)

            if idx >= 0: 
                self.run_one_clip(sliding_seg)
    
    def run_one_batch(self, data):
        for sliding_seg in data:
            step = self.current_step
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']

            # run one batch
            self.run_one_clip(sliding_seg)
            self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step)
            self.current_step = self.current_step + 1
            self.post_processing.init_flag = False