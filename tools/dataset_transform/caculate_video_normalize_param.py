'''
Author       : Thyssen Wen
Date         : 2024-01-09 14:08:33
LastEditors  : Thyssen Wen
LastEditTime : 2024-01-20 15:01:08
Description  : file content
FilePath     : /SVTAS/tools/dataset_transform/caculate_video_normalize_param.py
'''
import argparse

import re
import ffmpy
import numpy as np
import argparse
import os
import os.path as osp
import decord as de
from tqdm import tqdm
import random

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
    elif dataset_type in ['UBnormal']:
        file_ptr = open(input_path, 'r')
        info = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        refine_info = []
        for info_name in info:
            type, scene_id, clip_id = re.findall('(abnormal|normal)_scene_(\d+)_scenario(.*).*', info_name.split('.')[0])[0]
            file_name = os.path.join(f"Scene{scene_id}", info_name.split('.')[0])
            refine_info.append([info_name, file_name])
        info = refine_info
    return info

def load_file(videos_path, file_path, dataset_type):
    """Load index file to get video information."""
    video_path_list = []
    video_segment_lists = parse_file_paths(file_path, dataset_type)
    for video_segment in video_segment_lists:
        if dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            video_name = video_segment.split('.')[0]

            video_path = os.path.join(videos_path, video_name + '.mp4')
            video_path_list.append(video_path)

        elif dataset_type in ['breakfast', 'UBnormal']:
            video_segment_name, video_segment_path = video_segment
            video_name = video_segment_name.split('.')[0]

            video_path = os.path.join(videos_path, video_segment_path + '.mp4')
            video_path_list.append(video_path)
                
    return video_path_list

def main():
    args = get_arguments()
    video_path_list = load_file(args.videos_path, args.label_path, args.data_type)

    video_mean = []
    video_std = []
    for video_path in tqdm(video_path_list, desc="Calculate video info: "):
        video_capture = de.VideoReader(video_path)
        videolen = len(video_capture)
        
        # store mean std
        frames_select = random.sample(range(videolen - 1), 10)
        frames_select.sort()
        if len(frames_select) > 0:
            bgr_image = video_capture.get_batch(frames_select).asnumpy()
            # caculate BGR std and mean
            norm_imgs = np.reshape(bgr_image, (-1, 3))
            action_mean = list(np.mean(norm_imgs, axis=0))
            action_std = list(np.std(norm_imgs, axis=0))
            video_mean.append(action_mean)
            video_std.append(action_std)
        else:
            bgr_image = video_capture.get_batch([(videolen - 1)//2]).asnumpy()
            # caculate BGR std and mean
            norm_imgs = np.reshape(bgr_image, (-1, 3))
            action_mean = list(np.mean(norm_imgs, axis=0))
            action_std = list(np.std(norm_imgs, axis=0))
            video_mean.append(action_mean)
            video_std.append(action_std)

    total_mean = list(np.mean(np.array(video_mean), axis=0))
    total_std = list(np.mean(np.array(video_std), axis=0))
    print("mean RGB :", total_mean, "std RGB :", total_std)

def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="caculate video normalize info")
    parser.add_argument("label_path", type=str, help="path of a label files")
    parser.add_argument("videos_path", type=str, help="path of a video files")
    parser.add_argument(
        "--data_type",
        type=str,
        help="path of output localization label json.",
        default="gtea"
    )

    return parser.parse_args()

if __name__ == "__main__":
    main()