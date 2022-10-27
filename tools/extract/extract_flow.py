'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:37:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:14:47
Description  : Extract flow script
FilePath     : /SVTAS/tools/extract/extract_flow.py
'''

import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import torch
import numpy as np
import cv2
import svtas.model.builder as model_builder
import argparse
from svtas.utils.logger import get_logger, setup_logger
from svtas.utils.config import Config
from tqdm import tqdm
import decord
import svtas.loader.builder as dataset_builder
from PIL import Image
from svtas.utils.flow_vis import make_colorwheel
from svtas.runner.extract_runner import ExtractRunner

@torch.no_grad()
def extractor(cfg, args):
    out_path = os.path.join(args.outpath, "flow")
    isExists = os.path.exists(out_path)
    if not isExists:
        os.makedirs(out_path)
        print(out_path + ' created successful')
    logger = get_logger("SVTAS")
    
    # construct model
    model = model_builder.build_model(cfg.FLOW_MODEL).cuda()

    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 1)

    assert video_batch_size == 1, "Only support 1 batch size"

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
    
    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = ExtractRunner(logger=logger, model=model, post_processing=post_processing, feature_out_path=out_path, logger_interval=cfg.get('logger_interval', 100))

    runner.epoch_init()
    for i, data in enumerate(dataloader):
        runner.run_one_iter(data=data)
    
    logger.info("Finish all extracting!")

@torch.no_grad()
def extractor_aaa(cfg, outpath, need_visualize):
    model = model_builder.build_model(cfg.MODEL).cuda()
    transforms = VideoStreamTransform(cfg.TRANSFORM)
    post_transforms = VideoStreamTransform([dict(Clamp = dict(min_val=-20, max_val=20)),
                                            dict(ToUInt8 = None)])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    model.eval()

    out_path = os.path.join(outpath, "flow")
    isExists = os.path.exists(out_path)
    if not isExists:
        os.makedirs(out_path)
        print(out_path + ' created successful')
    
    if need_visualize:
        video_out_path = os.path.join(outpath, "flow_video")
        isExists = os.path.exists(video_out_path)
        if not isExists:
            os.makedirs(video_out_path)
            print(video_out_path + ' created successful')


    for file in tqdm(file_list, desc="extract optical flow"):
        # load video
        video_name = file.split('.')[0]
        vid_path = os.path.join(cfg.DATASET.video_path, video_name + '.mp4')
        if not os.path.isfile(vid_path):
            vid_path = os.path.join(cfg.DATASET.video_path, video_name + '.avi')
            if not os.path.isfile(vid_path):
                raise NotImplementedError
                
        video = decord.VideoReader(vid_path)
        frames_select = video.get_batch([0])
        img = frames_select.asnumpy()
        imgbuf = img[0].copy()
        img = Image.fromarray(imgbuf, mode='RGB')
        video_len = len(video)
        # store flow img
        flow_img_list = []

        if need_visualize:
            flow_video_path = os.path.join(video_out_path, video_name + '.mp4')
            videoWrite = cv2.VideoWriter(flow_video_path, fourcc, cfg.DATASET.fps, img.size)

        for start_frame in range(0, video_len, cfg.DATASET.num_segments):
            end_frame = start_frame + cfg.DATASET.num_segments
            if end_frame > video_len:
                end_frame = video_len
            frames_idx = list(range(start_frame, end_frame))
            frames_select = video.get_batch(frames_idx)
            imgs = []
            # dearray_to_img
            np_frames = frames_select.asnumpy()
            for i in range(np_frames.shape[0]):
                imgbuf = np_frames[i].copy()
                imgs.append(Image.fromarray(imgbuf, mode='RGB'))

            input_data = {}
            input_data['imgs'] = imgs
            input_data = transforms(input_data)

            imgs = input_data['imgs']
            imgs = imgs.unsqueeze(0).cuda()
            input_data['imgs'] = imgs
            flows = model(input_data).squeeze(0)

            results = {}
            results['imgs'] = flows
            flows = post_transforms(results)['imgs']
            flows = flows.cpu().permute(0, 2, 3, 1).numpy()
            flow_img_list.append(flows)

            if need_visualize:
                u = flows[:, :, :, 0]
                v = flows[:, :, :, 1]
                rad = np.sqrt(np.square(u) + np.square(v))
                rad_max = np.max(rad)
                epsilon = 1e-5
                u = u / (rad_max + epsilon)
                v = v / (rad_max + epsilon)
                
                colorwheel = make_colorwheel()  # shape [55x3]
                flows_image = np.zeros((u.shape[0], u.shape[1], u.shape[2], 3), np.uint8)

                ncols = colorwheel.shape[0]
                rad = np.sqrt(np.square(u) + np.square(v))
                a = np.arctan2(-v, -u)/np.pi
                fk = (a + 1) / 2 * (ncols - 1)
                k0 = np.floor(fk).astype(np.int32)
                k1 = k0 + 1
                k1[k1 == ncols] = 0
                f = fk - k0
                for i in range(colorwheel.shape[1]):
                    tmp = colorwheel[:, i]
                    col0 = tmp[k0] / 255.0
                    col1 = tmp[k1] / 255.0
                    col = (1 - f) * col0 + f * col1
                    idx = (rad <= 1)
                    col[idx]  = 1 - rad[idx] * (1-col[idx])
                    col[~idx] = col[~idx] * 0.75   # out of range
                    # Note the 2-i => BGR instead of RGB
                    ch_idx = 2 - i
                    flows_image[:, :, :, ch_idx] = np.floor(255 * col)

                for flow_img in flows_image:
                    videoWrite.write(flow_img)
                
            model._clear_memory_buffer()
        # save flow imgs
        flow_imgs = np.concatenate(flow_img_list, axis=0)
        save_path = os.path.join(out_path, video_name + '.npy')
        np.save(save_path, flow_imgs)
        
        if need_visualize:
            videoWrite.release()

def parse_args():
    parser = argparse.ArgumentParser("SVTAS extract optical flow script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    parser.add_argument("--need_visualize",
                        action="store_true",
                        help="wheather need optical flow visualization video")
    parser.add_argument("--need_feature",
                        action="store_true",
                        help="wheather need optical flow visualization video")
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    setup_logger(f"./output/etract_flow", name="SVTAS", level="INFO", tensorboard=False)
    extractor(cfg, args)

if __name__ == '__main__':
    main()