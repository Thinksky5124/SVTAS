'''
Author       : Thyssen Wen
Date         : 2022-10-23 10:27:54
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-20 10:56:46
Description  : Use Grad-CAM to visualization Video Infer Process ref:https://github.com/jacobgil/pytorch-grad-cam
FilePath     : /SVTAS/tools/visualize/cam_visualization.py
'''
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import argparse
import torch
from svtas.utils.config import Config
import svtas.model.builder as model_builder
import svtas.loader.builder as dataset_builder
from mmengine.runner import load_state_dict
from svtas.utils.logger import get_logger, setup_logger
from svtas.utils.cam import ModelForwardWrapper, get_match_fn_class
from svtas.engine.visual_engine import VisualRunner

from pytorch_grad_cam import GradCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad
        
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
    model = ModelForwardWrapper(model=model, data_key=cfg.VISUALIZE.data_key, sample_rate=cfg.sample_rate)
    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE.test)
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
    
    batch_infer = False
    if cfg.COLLATE.test.name in ["BatchCompose"]:
        batch_infer = True
        
    visualize_cfg = cfg.VISUALIZE
    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)
    match_fn = get_match_fn_class(cfg.VISUALIZE.match_fn_name)

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
                 methods=methods,
                 match_fn=match_fn)

    runner.epoch_init()
    dataloader.dataset._viodeo_sample_shuffle()
    for i, data in enumerate(dataloader):
        if batch_infer is True:
            runner.run_one_batch(data=data)
        elif len(data) == temporal_clip_batch_size or len(data[0]['labels'].shape) != 0:
            runner.run_one_iter(data=data)
        else:
            break
    
    logger.info("Finish all visualization!")