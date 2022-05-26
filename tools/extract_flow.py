'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:37:08
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-26 19:52:12
Description  : Extract flow script
FilePath     : /ETESVS/tools/extract_flow.py
'''

import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import torch
import numpy as np
import cv2
import model.builder as model_builder
import argparse
from utils.logger import get_logger, setup_logger
from utils.config import parse_config
from tqdm import tqdm
import decord
from loader.transform import VideoStreamTransform
from PIL import Image
from utils.flow_vis import make_colorwheel

@torch.no_grad()
def extractor(cfg, file_list, outpath):
    model = model_builder.build_model(cfg.MODEL).cuda()
    transforms = VideoStreamTransform(cfg.TRANSFORM)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    model.eval()

    out_path = os.path.join(outpath, "flow")
    isExists = os.path.exists(out_path)
    if not isExists:
        os.makedirs(out_path)
        print(out_path + ' created successful')


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
        flow_path = os.path.join(out_path, video_name + '.mp4')
        videoWrite = cv2.VideoWriter(flow_path, fourcc, cfg.DATASET.fps, img.size)

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

            flows = flows.cpu().permute(0, 2, 3, 1).numpy()

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
        videoWrite.release()
        

def parse_args():
    parser = argparse.ArgumentParser("ETESVS extract optical flow script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args

def parse_file_paths(input_path, dataset_type):
        if dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            file_ptr = open(input_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
        elif dataset_type in ['breakfast']:
            file_ptr = open(input_path, 'r')
            info = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            refine_info = []
            for info_name in info:
                video_ptr = info_name.split('.')[0].split('_')
                file_name = ''
                for j in range(2):
                    if video_ptr[j] == 'stereo01':
                        video_ptr[j] = 'stereo'
                    file_name = file_name + video_ptr[j] + '/'
                file_name = file_name + video_ptr[2] + '_' + video_ptr[3]
                if 'stereo' in file_name:
                    file_name = file_name + '_ch0'
                refine_info.append([info_name, file_name])
            info = refine_info
        return info
        
def main():
    args = parse_args()
    cfg = parse_config(args.config)
    setup_logger(f"./output/etract_flow", name="ETESVS", level="INFO", tensorboard=False)
    file_list = parse_file_paths(cfg.DATASET.file_list, cfg.DATASET.dataset_type)
    extractor(cfg, file_list, args.out_path)

if __name__ == '__main__':
    main()