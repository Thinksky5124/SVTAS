'''
Author       : Thyssen Wen
Date         : 2022-05-10 10:15:26
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-08 20:10:39
Description  : statistic labels number for dataset
FilePath     : /SVTAS/tools/data_anlysis/statistic_labels_num.py
'''
import argparse
import os
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="convert pred and gt list to images.")
    parser.add_argument(
        "file_list_path",
        type=str,
        help="path to dataset file list",
    )
    parser.add_argument(
        "labels_path",
        type=str,
        help="path to dataset labels",
    )
    parser.add_argument(
        "mapping_txt_path",
        type=str,
        help="path to mapping labels",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="path to output img",
        default="output"
    )

    return parser.parse_args()

def load_action_dict(label_path):
    with open(label_path, "r", encoding='utf-8') as f:
        actions = f.read().split("\n")[:-1]

    id2class_map = dict()
    class2id_map = dict()
    for a in actions:
        id2class_map[int(a.split(" ")[0])] = a.split(" ")[1]
        class2id_map[a.split()[1]] = int(a.split()[0])

    return id2class_map, class2id_map

def parse_file_paths(input_path):
    file_ptr = open(input_path, 'r')
    info = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    return info
        
def main() -> None:
    args = get_arguments()
    
    file_list = parse_file_paths(args.file_list_path)
    id2class_map, class2id_map = load_action_dict(args.mapping_txt_path)

    num_dict = {}
    total_frame_cnt = {}
    duration_list = []

    nums = [0 for i in range(len(id2class_map))]
    boundary_nums = [0 for i in range(2)]
    frames_cnt = 0 
    for file_name in tqdm(file_list, desc="label count"):
        video_name = file_name.split('.')[0]
        label_path = os.path.join(args.labels_path, video_name + '.txt')
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]

        gt_array = np.zeros(len(content))
        boundary_array = np.zeros(len(content), dtype=np.int64)
        last = content[0]
        boundary_array[0] = 1
        for i in range(len(content)):
            gt_array[i] = class2id_map[content[i]]
            if i > 0 :
                if last != content[i]:
                    boundary_array[i] = 1
                    last = content[i]

        frames_cnt += len(content)
        gt_array = gt_array.astype(np.int64)
        num, cnt = np.unique(gt_array, return_counts=True)
        for n, c in zip(num, cnt):
            nums[n] += c
        
        num, cnt = np.unique(boundary_array, return_counts=True)
        for n, c in zip(num, cnt):
            boundary_nums[n] += c
        
        # save action duration
        boundary_index_list = [0]
        before_action_name = content[0]
        for index in range(1, len(content)):
            if before_action_name != content[index]:
                boundary_index_list.append(index)
                before_action_name = content[index]
        boundary_index_list.append(len(content) - 1)
        for index in range(len(boundary_index_list) - 1):
            start_frame = float(boundary_index_list[index])
            end_frame = float(boundary_index_list[index + 1] - 1)
            duration_list.append(end_frame - start_frame)

        count_dict = pd.value_counts(content)

        for key, value in count_dict.items():
            if key not in num_dict.keys():
                num_dict[key] = value
                total_frame_cnt[key] = len(content)
            else:
                num_dict[key] = num_dict[key] + value
                total_frame_cnt[key] = total_frame_cnt[key] + len(content)
    
    num_duraction = np.array(duration_list)
    
    plt.hist(num_duraction, bins=100, density = True, range=(0, 400))
    plt.vlines(64, 0, 0.008, color="red")
    plt.title('hist for action duration')
    plt.xlabel("Frame duration")
    plt.ylabel('Density')
    plt.savefig(os.path.join(args.output_dir, "action_duration_count.png"), bbox_inches='tight', dpi=500)
    plt.close()
    
    print(num_dict)

    names = list(num_dict.keys())
    x = range(len(names))
    y = list(num_dict.values())
    plt.bar(x, y)
    plt.xticks(x, names, rotation=90)
    plt.title('Categories of statistical')
    plt.xlabel("Labels' name")
    plt.ylabel('Number')
    plt.xticks(fontsize=5)
    plt.savefig(os.path.join(args.output_dir, "labels_count.png"), bbox_inches='tight', dpi=500)
    plt.close()

    weights_dict = {}
    # crossentropy weight compute by median frequency balancing
    """
    Class weight for CrossEntropy
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """
    class_num = np.array(nums, dtype=np.float32)
    total = class_num.sum().item()
    frequency = class_num / total
    median = np.median(frequency)
    class_weight = median / frequency

    for i in range(len(id2class_map)):
        weights_dict[i] = class_weight[i]
    
    """
    pos_weight for binary cross entropy with logits loss
    pos_weight is defined as reciprocal of ratio of positive samples in the dataset
    """
    pos_ratio = boundary_nums / sum(boundary_nums)
    pos_weight = 1 / pos_ratio

    out_txt_file_path = os.path.join(args.output_dir, "weights.txt")
    f = open(out_txt_file_path, "w", encoding='utf-8')
    f.write("Class weight for CrossEntropy: \n")
    for key, action_wights in weights_dict.items():
        str_str = str(key) + " " + str(action_wights) + "\n"
        f.write(str_str)
    f.write("Position weight for Boundary BCEWithLogitsLoss: \n")
    for name, weights in zip(["unboundary", "boundary"], pos_weight):
        str_str = str(name) + " " + str(weights) + "\n"
        f.write(str_str)
    f.close()


if __name__ == "__main__":
    main()