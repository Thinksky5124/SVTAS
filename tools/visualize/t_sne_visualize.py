'''
Author       : Thyssen Wen
Date         : 2022-11-15 16:10:58
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 15:16:22
Description  : file content
FilePath     : /SVTAS/tools/visualize/t_sne_visualize.py
'''
import argparse
from svtas.utils import is_matplotlib_available
if is_matplotlib_available():
    import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import os
import cv2

def tsne(args):
    if args.input.endswith("mp4") or args.input.endswith("avi"):
        imgs_list = []
        video = cv2.VideoCapture(args.input)
        while True:
            ret, frame = video.read()
            if ret:
                imgs_list.append(np.expand_dims(frame, axis=0))
            else:
                break
        feature = np.concatenate(imgs_list, axis=0)
        feature = feature.reshape((feature.shape[0], -1)).T
    else:
        with open(args.input,"rb") as input_file:
            feature = np.load(input_file)

    with open(args.label,"r") as label_file:
        labels = label_file.read().split('\n')[:-1]

    # 选择维度
    tsne = TSNE(n_components=2)
    tsne_obj= tsne.fit_transform(feature.T)

    tsne_df = pd.DataFrame({'X':tsne_obj[:,0],
                            'Y':tsne_obj[:,1],
                            'label':labels})
    
    img = sns.scatterplot(x="X", y="Y",
                hue="label",
                # marker="|",
                data=tsne_df)
    img.legend(ncol=4, fontsize = 5)
    img.figure.savefig(os.path.join(args.out_path, "t-SNE_visalize.png"), bbox_inches='tight', dpi=500)
    plt.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str,
                        default='input.npy',
                        help='input feature file to visualize')
    parser.add_argument('-l',
                        '--label',
                        type=str,
                        default='video.txt',
                        help='label txt file for every frame')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        default='./output',
                        help='visaulzie file out path')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    tsne(args)