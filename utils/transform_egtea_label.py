'''
Author: Thyssen Wen
Date: 2022-04-13 18:33:33
LastEditors: Thyssen Wen
LastEditTime: 2022-04-26 21:30:34
Description: transform egtea dataset label script
FilePath: /ETESVS/utils/transform_egtea_label.py
'''
from __future__ import annotations
from asyncore import write
from cgi import test
import json
import numpy as np
import argparse
import os
import decord as de

from tqdm import tqdm

def load_action_dict(label_path):
    with open(label_path, "r", encoding='utf-8') as f:
        actions = f.read().split("\n")[:-1]

    id2class_map = dict()
    for a in actions:
        id2class_map[int(a.split(" ")[1])] = a.split(" ")[0]

    return id2class_map

def generate_mapping_list_txt(action_dict, out_path):
    out_txt_file_path = os.path.join(out_path, "mapping.txt")
    f = open(out_txt_file_path, "w", encoding='utf-8')
    for key, action_name in action_dict.items():
        str_str = str(key) + " " + action_name + "\n"
        f.write(str_str)
    # add None
    str_str = str(len(action_dict)) + " None" + "\n"
    f.write(str_str)
    f.close()
    
def main():
    args = get_arguments()
    fils_name = os.listdir(args.label_path)

    id2class_map = load_action_dict(args.action_idx_path)

    generate_mapping_list_txt(id2class_map, args.out_path)

    test_list = []
    train_list = []
    for file in fils_name:
        if file.startswith("test"):
            test_list.append(file)
        elif file.startswith("train"):
            train_list.append(file)

    split_dict = {"version": "EGTEA", "fps":args.fps, "database":{}}
    for i in tqdm(range(len(test_list)), desc="convert"):
        test_split, train_split = test_list[i], train_list[i]
        test_file = open(os.path.join(args.label_path, test_split), "r", encoding='utf-8')
        test_split_video_name_list = []
        train_split_video_name_list = []
        train_file = open(os.path.join(args.label_path, train_split), "r", encoding='utf-8')
        for line in test_file:
            info = line.strip().split(" ")
            video_name_frame_start_end_list = info[0].split("-")
            verb_label_name = id2class_map[int(info[2])]
            video_name = '-'.join(video_name_frame_start_end_list[:3])
            start_second = float(video_name_frame_start_end_list[5][1:]) / args.fps
            end_second = float(video_name_frame_start_end_list[6][1:]) / args.fps

            if video_name not in list(split_dict["database"].keys()):
                # get frames
                video_path = os.path.join(args.out_path, "Videos", video_name + ".mp4")
                video_len = len(de.VideoReader(video_path))

                split_dict["database"][video_name] = {"subset":"test", "frames": video_len, "annotations":[
                    {"segment": [start_second, end_second], "label": verb_label_name}
                ]}
            else:
                find_flag = False
                for annotation in split_dict["database"][video_name]["annotations"]:
                    if annotation["label"] == verb_label_name and \
                        abs(annotation["segment"][0] - start_second) < 1 and \
                            abs(annotation["segment"][1] - end_second) < 1:
                            find_flag = True
                    if find_flag is True:
                        break
                if find_flag is False:
                    split_dict["database"][video_name]["annotations"].append(
                        {"segment": [start_second, end_second], "label": verb_label_name})
            
            if video_name + ".txt" not in test_split_video_name_list:
                test_split_video_name_list.append(video_name + ".txt")
        
        for line in train_file:
            info = line.strip().split(" ")
            video_name_frame_start_end_list = info[0].split("-")
            verb_label_name = id2class_map[int(info[2])]
            video_name = '-'.join(video_name_frame_start_end_list[:3])
            start_second = float(video_name_frame_start_end_list[5][1:]) / args.fps
            end_second = float(video_name_frame_start_end_list[6][1:]) / args.fps

            if video_name not in list(split_dict["database"].keys()):
                # get frames
                video_path = os.path.join(args.out_path, "Videos", video_name + ".mp4")
                video_len = len(de.VideoReader(video_path))
                
                split_dict["database"][video_name] = {"subset":"validation", "frames": video_len, "annotations":[
                    {"segment": [start_second, end_second], "label": verb_label_name}
                ]}
            else:
                find_flag = False
                for annotation in split_dict["database"][video_name]["annotations"]:
                    if annotation["label"] == verb_label_name and \
                        abs(annotation["segment"][0] - start_second) < 1e-1 and \
                            abs(annotation["segment"][1] - end_second)< 1e-1:
                            find_flag = True
                if find_flag is False:
                    split_dict["database"][video_name]["annotations"].append(
                        {"segment": [start_second, end_second], "label": verb_label_name})
            
            if video_name + ".txt" not in train_split_video_name_list:
                train_split_video_name_list.append(video_name + ".txt")
        
        out_test_txt_file_path = os.path.join(args.out_path, "splits", test_split)
        out_train_txt_file_path = os.path.join(args.out_path, "splits", train_split)
        str = '\n'
        f = open(out_test_txt_file_path, "w", encoding='utf-8')
        f.write(str.join(test_split_video_name_list) + str)
        f.close()
        f = open(out_train_txt_file_path, "w", encoding='utf-8')
        f.write(str.join(train_split_video_name_list) + str)
        f.close()

    write_path = os.path.join(args.out_path, "egtea" + ".json")
    with open(write_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=4)


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert segmentation and localization label")
    parser.add_argument("label_path", type=str, help="path of a label files")
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
    parser.add_argument(
        "--fps",
        type=float,
        default=24.0,
        help="Convert label fps.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    main()
