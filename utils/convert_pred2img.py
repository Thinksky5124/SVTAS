'''
Author: Thyssen Wen
Date: 2022-03-25 16:44:12
LastEditors: Thyssen Wen
LastEditTime: 2022-04-08 17:36:08
Description: convert img function script
FilePath: /ETESVS/utils/convert_pred2img.py
'''
import argparse
import os
from tqdm import tqdm

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


def get_arguments() -> argparse.Namespace:
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(description="convert pred and gt list to images.")
    parser.add_argument(
        "input_dir",
        type=str,
        help="path to a files you want to convert",
    )
    parser.add_argument(
        "action_dict_path",
        type=str,
        help="path to a action dict file",
    )
    parser.add_argument(
        "output_dir",
        type=str,
        help="path to output img",
        default="output"
    )
    parser.add_argument(
        "--sliding_windows",
        type=int,
        help="sliding windows szie",
        default=120
    )

    return parser.parse_args()


def convert_arr2img(file_path, palette, actions_dict):
    """
    Args:
        str: file path
        palette: color palette
    """
    file_ptr = open(file_path, 'r')
    list = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    
    array = np.array([actions_dict[name] for name in list])

    arr = array.astype(np.uint8)
    arr = np.tile(arr, (100, 1))
    
    return arr

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit


    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def main() -> None:
    args = get_arguments()
    action_dict_path = args.action_dict_path

    # actions dict generate
    file_ptr = open(action_dict_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    palette = make_palette(len(actions_dict))

    isExists = os.path.exists(args.output_dir)
    if not isExists:
        os.makedirs(args.output_dir)

    filenames = os.listdir(args.input_dir)

    vid_list = ["-".join(vid.split('-')[:-1]) for vid in filenames if vid.endswith('pred.txt')]

    for vid in tqdm(vid_list, desc='label convert:'):
        gt_file_path = os.path.join(args.input_dir, vid + '-gt.txt')
        pred_file_path = os.path.join(args.input_dir, vid + '-pred.txt')
        gt_arr = convert_arr2img(gt_file_path, palette, actions_dict)
        pred_arr = convert_arr2img(pred_file_path, palette, actions_dict)

        arr = np.concatenate([gt_arr, pred_arr], axis=0)
        img = Image.fromarray(arr)
        img = img.convert("P")
        img.putpalette(palette)

        plt.figure(vid)
        plt.title('GroundTruth')
        plt.imshow(img)
        plt.xlabel('Prediction')
        plt.gca().xaxis.set_major_locator(MultipleLocator(args.sliding_windows))
        
        plt.savefig(os.path.join(args.output_dir, vid + ".png"), bbox_inches='tight', dpi=500)

if __name__ == "__main__":
    main()