'''
Author       : Thyssen Wen
Date         : 2023-02-23 10:16:29
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-23 12:03:48
Description  : file content
FilePath     : /SVTAS/tools/dataset_transform/video_coding_transform.py
'''
import cv2
import ffmpy
import numpy as np
import argparse
import os
import os.path as osp
from tqdm import tqdm

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

def load_file(videos_path, file_path, dataset_type):
    """Load index file to get video information."""
    video_path_list = []
    video_segment_lists = parse_file_paths(file_path, dataset_type)
    for video_segment in video_segment_lists:
        if dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
            video_name = video_segment.split('.')[0]

            video_path = os.path.join(videos_path, video_name + '.avi')
            video_path_list.append(video_path)

        elif dataset_type in ['breakfast']:
            video_segment_name, video_segment_path = video_segment
            video_name = video_segment_name.split('.')[0]

            video_path = os.path.join(videos_path, video_segment_path + '.avi')
            video_path_list.append(video_path)
                
    return video_path_list

def main():
    args = get_arguments()
    video_path_list = load_file(args.videos_path, args.label_path, args.data_type)

    for video_path in tqdm(video_path_list, desc="Transform video to mp4: "):
        file_dir_path = video_path.split('/')[:-1]
        video_name = video_path.split('/')[-1].split('.')[0]
        dump_tem_file = os.path.join(".")
        for dir_path in file_dir_path:
            if dir_path == 'Videos':
                dump_tem_file = os.path.join(dump_tem_file, 'Videos_mp4')
            else:
                dump_tem_file = os.path.join(dump_tem_file, dir_path)

        folder = os.path.exists(dump_tem_file)
        if not folder:
            os.makedirs(dump_tem_file)
        dump_tem_file = os.path.join(dump_tem_file, video_name + '.mp4')
        
        ff = ffmpy.FFmpeg(
            inputs={video_path: None},
            outputs={dump_tem_file: None}
        )

        ff.run()

def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert fine segmentation label")
    parser.add_argument("label_path", type=str, help="path of a label files")
    parser.add_argument("videos_path", type=str, help="path of a video files")
    parser.add_argument(
        "--data_type",
        type=str,
        help="path of output localization label json.",
        default="gtea"
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether to use resume transform')

    return parser.parse_args()

if __name__ == "__main__":
    main()