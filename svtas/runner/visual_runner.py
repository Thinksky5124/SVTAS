'''
Author       : Thyssen Wen
Date         : 2022-10-31 19:02:43
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 19:13:33
Description  : file content
FilePath     : /SVTAS/svtas/runner/visual_runner.py
'''
import os
import cv2
import queue
import numpy as np
import torch
import copy

from tools.infer.infer import make_palette, label_arr2img, draw_action_label
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    preprocess_image
from pytorch_grad_cam.ablation_layer import AblationLayerVit

def reshape_transform(transform_form):
# # class activation transform [N C T]
    def reshape_transform_NCT(tensor, height=1, width=1):
        result = torch.reshape(tensor, [tensor.shape[0], tensor.shape[1], height, width])
        result = torch.permute(result, [0, 2, 3, 1])

        # Bring the channels to the first dimension,
        # like in CNNs.
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # feature activation transform [N P C]
    def reshape_transform_NPT(tensor, height=7, width=7):
        result = tensor[:, 1:, :].reshape(tensor.size(0),
                                        height, width, tensor.size(2))

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
        return reshape_transform_NPT
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
                 methods):
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
            self.targets = self.visualize_cfg.return_targets_name

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
        frame_height = self.visualize_cfg.output_frame_size[1]
        frame_width = self.visualize_cfg.output_frame_size[0]
        for cam_imgs, vid, labels, preds in zip(cam_imgs_list, current_vid_list, labels_list, preds_list):
            cam_imgs_save_path = os.path.join(self.cam_imgs_out_path, vid + ".mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            video = cv2.VideoWriter(cam_imgs_save_path, fourcc, self.visualize_cfg.fps, (frame_width, frame_height))
            pred_queue = queue.Queue(maxsize=32)
            label_queue = queue.Queue(maxsize=32)
            for idx in range(cam_imgs.shape[0]):
                img = cam_imgs[idx]
                img = cv2.resize(img, (frame_width, frame_height))
                # add pred and gt info
                cv2.putText(img, "Prediction: " + self.actions_dict[preds[idx]], (0, frame_height - 100), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                cv2.putText(img, "Groundtruth: " + self.actions_dict[labels[idx]], (0, frame_height - 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                if pred_queue.full():
                    pred_queue.get()
                    label_queue.get()
                pred_queue.put([preds[idx]])
                label_queue.put([labels[idx]])
                pred_img = label_arr2img(pred_queue, self.palette).convert('RGB')
                label_img = label_arr2img(label_queue, self.palette).convert('RGB')
                past_width = int((label_img.size[0] / 32) * (frame_width - 40))
                pred_img = cv2.cvtColor(np.asarray(pred_img),cv2.COLOR_RGB2BGR)
                label_img = cv2.cvtColor(np.asarray(label_img),cv2.COLOR_RGB2BGR)
                pred_img = cv2.resize(pred_img, (past_width, 20))
                label_img = cv2.resize(label_img, (past_width, 20))
                cv2.putText(img, "Pr: ", (0, frame_height - 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                img[(frame_height - 50):(frame_height - 30), 30:(30 + past_width), :] = pred_img
                cv2.putText(img, "GT: ", (0, frame_height - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                img[(frame_height - 30):(frame_height - 10), 30:(30 + past_width), :] = label_img
                # Line 1 prediction Line 2 groundtruth
                img = cv2.rectangle(img, (20 + past_width, frame_height - 10), (30 + past_width, frame_height - 50), (255, 255, 255), thickness=-1)
                cv2.line(img, (30, frame_height - 30), (30 + past_width, frame_height - 30), (255,255,255), 1)
                cv2.putText(img, "Current Frame", (max(past_width - 110, 0), frame_height - 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                data_pred = list(copy.deepcopy(pred_queue.queue))
                data_label = list(copy.deepcopy(label_queue.queue))
                array_pred = np.array(data_pred).transpose()
                array_label = np.array(data_label).transpose()
                label = list(set(array_pred[0, :].tolist()) | set(array_label[0, :].tolist()))
                img = draw_action_label(img, self.palette, self.actions_dict, label)
                video.write(img)
            video.release()

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
        input_tensor = input_data['imgs'].reshape([-1]+list(input_data['imgs'].shape[-3:]))
        with torch.no_grad():
            outputs = self.model(input_tensor)
            score = outputs['output']

        grayscale_cam = self.cam(input_tensor=input_tensor,
                            targets=self.targets,
                            eigen_smooth=self.eigen_smooth,
                            aug_smooth=self.aug_smooth)

        # Here grayscale_cam has only one image in the batch
        cam_image_list = []
        for batch_id in range(len(data_dict['raw_imgs'])):
            batch_image_list = []
            for sample_id in range(len(data_dict['raw_imgs'][batch_id])):
                rgb_img = cv2.cvtColor(np.asarray(data_dict['raw_imgs'][batch_id][sample_id]), cv2.COLOR_RGB2BGR)[:, :, ::-1]
                rgb_img = np.float32(rgb_img) / 255
                rgb_img = cv2.resize(rgb_img, (grayscale_cam.shape[-1], grayscale_cam.shape[-2]))
                grayscale_cam_sample = grayscale_cam[batch_id * len(data_dict['raw_imgs'][batch_id]) + sample_id, :]

                cam_image = show_cam_on_image(rgb_img, grayscale_cam_sample)
                batch_image_list.append(np.expand_dims(cam_image, 0))
            batch_image = np.expand_dims(np.concatenate(batch_image_list, 0), 0)
            cam_image_list.append(batch_image)
        cam_images = np.concatenate(cam_image_list, 0)
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