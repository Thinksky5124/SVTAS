'''
Author       : Thyssen Wen
Date         : 2022-07-18 16:24:00
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-18 20:03:24
Description  : file conten
FilePath     : /ETESVS/tools/transform_breakfast_fine_label.py
'''
'''
Author: Thyssen Wen
Date: 2022-04-13 18:33:33
LastEditors: Thyssen Wen
LastEditTime: 2022-04-27 10:13:08
Description: transform breakfast fine dataset label script
FilePath: /ETESVS/utils/transform_egtea_label.py
'''
import json
import argparse
import os

from tqdm import tqdm

def generate_mapping_list_txt(action_dict, out_path):
    out_txt_file_path = os.path.join(out_path, "mapping_fine.txt")
    f = open(out_txt_file_path, "w", encoding='utf-8')
    for key, action_name in action_dict.items():
        str_str = str(key) + " " + action_name + "\n"
        f.write(str_str)
    # add background
    str_str = str(len(action_dict)) + " background" + "\n"
    f.write(str_str)
    f.close()
    
def main():
    args = get_arguments()

    Filelist = []
    for home, dirs, files in os.walk(args.label_path):
        for filename in files:
            if filename.endswith("txt"):
                Filelist.append(os.path.join(home, filename))
    
    VideoFileset = set([])
    for home, dirs, files in os.walk(args.video_path):
        for filename in files:
            if filename.endswith("avi"):
                VideoFileset.add(os.path.join(home, filename))

    action_set = set([])
    video_name_list = []
    for file_name in tqdm(Filelist, desc="convert"):
        with open(file_name, "r", encoding='utf-8') as f:
            seg_label = []

            for line in f:
                info_list = line.strip().split(" ")
                info_time = info_list[0].split("-")
                start_idx = int(info_time[0]) - 1
                end_idx = int(info_time[1]) - 1
            
                if info_list[1] == "garbage":
                    info_list[1] = "background"
                else:
                    if info_list[1] not in action_set:
                        action_set.add(info_list[1])
                
                seg_label = seg_label + [info_list[1]] * (end_idx - start_idx)
            
            video_ptr = file_name.split("/")[-1].split('.')[0].split('_')
            file_name_str = ''
            for j in range(2):
                if video_ptr[j] == 'stereo01':
                    video_ptr[j] = 'stereo'
                file_name_str = file_name_str + video_ptr[j] + '/'
            file_name_str = file_name_str + video_ptr[2] + '_' + video_ptr[3]
            if 'stereo' in file_name_str:
                file_name_str = file_name_str + '_ch0'
            video_path = os.path.join(args.video_path, file_name_str + '.avi')
            
            if video_path in VideoFileset:
                video_name_list.append(file_name.split("/")[-1])

            out_txt_file_path = os.path.join(args.out_path, file_name.split("/")[-1])
            str = '\n'
            f = open(out_txt_file_path, "w", encoding='utf-8')
            f.write(str.join(seg_label) + str)
            f.close()

    action_set_list = list(action_set)
    action_dict = {}
    for idx in range(len(action_set_list)):
        action_dict[idx] = action_set_list[idx]

    generate_mapping_list_txt(action_dict, args.action_idx_path)
    video_name_list.sort()

    out_txt_file_path = os.path.join(args.action_idx_path, "fine_video_list.txt")
    str = '\n'
    f = open(out_txt_file_path, "w", encoding='utf-8')
    f.write(str.join(video_name_list) + str)
    f.close()


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert fine segmentation label")
    parser.add_argument("label_path", type=str, help="path of a label files")
    parser.add_argument("video_path", type=str, help="path of a video files")
    parser.add_argument(
        "action_idx_path",
        type=str,
        help="path of action index txt path.",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="path of output localization label json.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
