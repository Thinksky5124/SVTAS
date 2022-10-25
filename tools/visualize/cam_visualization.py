'''
Author       : Thyssen Wen
Date         : 2022-10-23 10:27:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-25 15:56:45
Description  : Use Grad-CAM to visualization Video Infer Process ref:https://github.com/jacobgil/pytorch-grad-cam
FilePath     : /SVTAS/tools/visualize/cam_visualization.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import argparse
import cv2
import queue
import numpy as np
import torch
import copy
from PIL import Image
from types import MethodType 
from utils.config import Config
import model.builder as model_builder
import loader.builder as dataset_builder
from mmcv.runner import load_state_dict
from utils.logger import get_logger, setup_logger
from tools.visualize.cam_forward_fn import cam_forward
from tools.infer.infer import make_palette, label_arr2img, draw_action_label

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

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

class CAMPostProcessing():
    def __init__(self,
                 sample_rate,
                 ignore_index=-100):
        self.init_flag = False
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
    
    def init_scores(self):
        self.imgs_list = []
        self.labels_list = []
        self.score_lsit = []
        self.init_flag = True

    def update(self, cam_images, labels, score, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        self.labels_list.append(labels)
        self.imgs_list.append(cam_images)
        with torch.no_grad():
            pred = np.argmax(score.reshape([labels.shape[0], labels.shape[1], -1]).detach().cpu().numpy(), axis=-1)
            self.score_lsit.append(pred)

    def output(self):
        imags_list = []
        labels_list = []
        preds_list = []

        labels = np.concatenate(self.labels_list, axis=1)
        imags = np.concatenate(self.imgs_list, axis=1)
        preds = np.concatenate(self.score_lsit, axis=1)

        for bs in range(imags.shape[0]):
            index = np.where(labels[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [labels[bs].shape[-1]]) // self.sample_rate
            imags_list.append(imags[bs, :ignore_start, :])
            labels_list.append(labels[bs, ::self.sample_rate][:ignore_start])
            preds_list.append(preds[bs, ::self.sample_rate][:ignore_start])

        return imags_list, labels_list, preds_list

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
                 cam_imgs_out_path):
        self.model = model
        self.logger = logger
        self.visualize_cfg = visualize_cfg
        self.post_processing = post_processing
        self.cam_imgs_out_path = cam_imgs_out_path
        self.cam_method = cam_method
        self.use_cuda = use_cuda
        self.eigen_smooth = eigen_smooth
        self.aug_smooth = aug_smooth
    
    def epoch_init(self):
        self.target_layers = []
        # batch videos sampler
        for layer in self.model.named_modules():
            if layer[0] in set(visualize_cfg.layer_name):
                self.target_layers.append(layer[1])

        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        if self.cam_method == "ablationcam":
            self.cam = methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.visualize_cfg.reshape_transform),
                                    ablation_layer=AblationLayerVit())
        else:
            self.cam = methods[self.cam_method](model=self.model,
                                    target_layers=self.target_layers,
                                    use_cuda=self.use_cuda,
                                    reshape_transform=reshape_transform(self.visualize_cfg.reshape_transform))
        self.cam.batch_size = visualize_cfg.batch_size
        # If None, returns the map for the highest scoring category.
        # Otherwise, targets the requested category.
        if visualize_cfg.return_targets_name is None:
            self.targets = None
        else:
            self.targets = visualize_cfg.return_targets_name

        # load mapping label
        file_ptr = open(visualize_cfg.label_path, 'r')
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
            if not torch.is_tensor(outputs):
                outputs = outputs[-1]

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
        return outputs, cam_images
    
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

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    parser.add_argument('--aug_smooth', action='store_true',
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument(
        '--eigen_smooth',
        action='store_true',
        help='Reduce noise by taking the first principle componenet'
        'of cam_weights*activations')

    parser.add_argument(
        '--method',
        type=str,
        default='gradcam',
        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam')

    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args


if __name__ == '__main__':
    """ python vit_gradcam.py --image-path <path_to_image>
    Example usage of using cam-methods on a VIT network.
    """

    args = get_args()
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad}

    if args.method not in list(methods.keys()):
        raise Exception(f"method should be one of {list(methods.keys())}")
    setup_logger(f"./output/etract_feature", name="SVTAS", level="INFO", tensorboard=False)
    outpath = args.out_path
    out_path = os.path.join(outpath, "cam")
    isExists = os.path.exists(out_path)
    if not isExists:
        os.makedirs(out_path)
        print(out_path + ' created successful')
    logger = get_logger("SVTAS")

    cfg = Config.fromfile(args.config)
    # construct model
    model = model_builder.build_model(cfg.MODEL).cuda()
    pretrain_path = cfg.get('PRETRAINED', None)
    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path)
        state_dicts = checkpoint["model_state_dict"]
        load_state_dict(model, state_dicts, logger=logger)
        # model.load_state_dict(state_dicts)
    # override forawrd method
    model.forward = MethodType(cam_forward, model)
    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE)
    Pipeline = dataset_builder.build_pipline(cfg.PIPELINE)
    dataset_config = cfg.DATASET.config
    dataset_config['pipeline'] = Pipeline
    dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    dataset_config['video_batch_size'] = video_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset_builder.build_dataset(dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)
    
    visualize_cfg = cfg.VISUALIZE
    post_processing = CAMPostProcessing(sample_rate=visualize_cfg.sample_rate,
                                        ignore_index=visualize_cfg.ignore_index)

    if args.method not in methods:
        raise Exception(f"Method {args.method} not implemented")

    runner = VisualRunner(cam_method=args.method,
                 use_cuda=args.use_cuda,
                 eigen_smooth=args.eigen_smooth,
                 aug_smooth=args.aug_smooth,
                 logger=logger, 
                 model=model, 
                 visualize_cfg=visualize_cfg,
                 post_processing=post_processing,
                 cam_imgs_out_path=out_path)

    runner.epoch_init()
    for i, data in enumerate(dataloader):
        runner.run_one_iter(data=data)
    
    logger.info("Finish all visualization!")