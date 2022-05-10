'''
Author       : Thyssen Wen
Date         : 2022-05-10 10:15:26
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-10 10:34:50
Description  : statistic labels number for dataset
FilePath     : /ETESVS/tools/statistic_labels_num.py
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
        "output_dir",
        type=str,
        help="path to output img",
        default="output"
    )

    return parser.parse_args()

def parse_file_paths(input_path):
    file_ptr = open(input_path, 'r')
    info = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    return info

def main() -> None:
    args = get_arguments()
    
    file_list = parse_file_paths(args.file_list_path)

    num_dict = {}

    for file_name in tqdm(file_list, desc="label count"):
        video_name = file_name.split('.')[0]
        label_path = os.path.join(args.labels_path, video_name + '.txt')
        file_ptr = open(label_path, 'r')
        content = file_ptr.read().split('\n')[:-1]

        count_dict = pd.value_counts(content)

        for key, value in count_dict.items():
            if key not in num_dict.keys():
                num_dict[key] = value
            else:
                num_dict[key] = num_dict[key] + value
    
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

if __name__ == "__main__":
    main()