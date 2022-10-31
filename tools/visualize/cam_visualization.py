'''
Author       : Thyssen Wen
Date         : 2022-10-23 10:27:54
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-31 19:09:43
Description  : Use Grad-CAM to visualization Video Infer Process ref:https://github.com/jacobgil/pytorch-grad-cam
FilePath     : /SVTAS/tools/visualize/cam_visualization.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import argparse
import numpy as np
import torch
from types import MethodType 
from svtas.utils.config import Config
import svtas.model.builder as model_builder
import svtas.loader.builder as dataset_builder
from mmcv.runner import load_state_dict
from svtas.utils.logger import get_logger, setup_logger
from tools.visualize.cam_forward_fn import cam_forward
from svtas.runner.visual_runner import VisualRunner

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad

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
                 cam_imgs_out_path=out_path,
                 methods=methods)

    runner.epoch_init()
    for i, data in enumerate(dataloader):
        runner.run_one_iter(data=data)
    
    logger.info("Finish all visualization!")