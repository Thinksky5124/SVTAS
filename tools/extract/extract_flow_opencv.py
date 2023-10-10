'''
Author       : Thyssen Wen
Date         : 2022-05-04 14:37:08
LastEditors  : Thyssen Wen
LastEditTime : 2023-04-16 13:07:18
Description  : Extract flow script
FilePath     : /SVTAS/tools/extract/extract_flow_opencv.py
'''

import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import argparse
import cv2
import numpy as np

def extract_optical_flow(video, out_path, video_name, bound=15):
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    flow_video = cv2.VideoWriter(os.path.join(out_path, video_name + '.mp4'), fourcc, fps, (width, height))
    zero = np.zeros((width, height, 3), dtype=np.uint8)
    flow_video.write(zero)
    success, prev = video.read()
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    while success:
        success, curr = video.read()
        if success:
            curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

            optflow_model = cv2.optflow.createOptFlow_DeepFlow()
            flow = optflow_model.calc(prev, curr, None)
            
            flow = (flow + bound) * (255.0 / (2 * bound))
            flow = np.round(flow).astype(int)
            flow[flow >= 255] = 255
            flow[flow <= 0] = 0
            zero = np.zeros((flow.shape[0], flow.shape[1], 1), dtype=int)
            flow = np.concatenate([flow, zero], axis=-1)
            prev = curr
            flow = np.uint8(flow)
            flow_video.write(flow)
    flow_video.release()

def extractor(args):
    path_list = []
    video_name_list = []
    with open(args.input_list, 'r') as f:
        for id, line in enumerate(f):
            video_name = line.strip()
            path_list.append(video_name)
            video_name_list.append(video_name.split('/')[-1].split('.')[-2])
    
    for video_path, video_name in zip(path_list, video_name_list):
        video = cv2.VideoCapture(video_path)
        extract_optical_flow(video=video, out_path=args.out_path, video_name=video_name)
        print(f"finish extract {video_name}.")

    
    print("Finish all extracting!")

def parse_args():
    parser = argparse.ArgumentParser("SVTAS extract optical flow script")
    parser.add_argument('-i',
                        '--input_list',
                        type=str,
                        default='configs/video_list.txt',
                        help='extract files path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    extractor(args)

if __name__ == '__main__':
    main()