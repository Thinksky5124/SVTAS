'''
Author       : Thyssen Wen
Date         : 2022-11-22 10:37:07
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 23:47:42
Description  : file content
FilePath     : /SVTAS/svtas/model/post_processings/cam_post_processing.py
'''
import numpy as np
import torch
from svtas.utils import AbstractBuildFactory
from svtas.utils.fileio import CAMVideoStreamWriter, CAMImageStreamWriter
from .base_post_processing import BasePostProcessing

@AbstractBuildFactory.register('post_processing')
class CAMVideoPostProcessing(BasePostProcessing):
    def __init__(self,
                 sample_rate,
                 output_frame_size,
                 fps=15,
                 ignore_index=-100,
                 need_label=True):
        super().__init__()
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
        self.fps = fps
        self.frame_height = output_frame_size[1]
        self.frame_width = output_frame_size[0]
        self.need_label = need_label
    
    def init_scores(self, sliding_num, batch_size):
        self.imgs_list = []
        self.labels_list = []
        self.score_lsit = []
        self.init_flag = True

    def update(self, cams_dict, labels, idx):
        cam_images, score = cams_dict['cam_images'], cams_dict['output']
        # seg_scores [stage_num N C T]
        # gt [N T]
        score = torch.transpose(score[-1], -1, -2)
        self.labels_list.append(labels)
        with torch.no_grad():
            pred = np.argmax(score.detach().cpu().numpy(), axis=-1)
            self.score_lsit.append(pred)
        for bs in range(cam_images.shape[0]):
            if len(self.imgs_list) < (bs + 1):
                self.imgs_list.append(CAMVideoStreamWriter(self.fps, self.frame_height, self.frame_width, need_label=self.need_label))
            self.imgs_list[bs].stream_write(cam_images[bs])

    def output(self):
        imags_list = []
        labels_list = []
        preds_list = []

        labels = np.concatenate(self.labels_list, axis=1)
        preds = np.concatenate(self.score_lsit, axis=1)

        for bs in range(len(self.imgs_list)):
            index = np.where(labels[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [labels[bs].shape[-1]]) // self.sample_rate
            self.imgs_list[bs].dump()
            imags_list.append({"writer":self.imgs_list[bs], "len":ignore_start})
            labels_list.append(labels[bs, ::self.sample_rate][:ignore_start])
            preds_list.append(preds[bs, ::self.sample_rate][:ignore_start])

        return imags_list, labels_list, preds_list

@AbstractBuildFactory.register('post_processing')
class CAMImagePostProcessing():
    def __init__(self,
                 sample_rate,
                 output_frame_size,
                 ignore_index=-100,
                 need_label=True,
                 need_split=False):
        self.init_flag = False
        self.ignore_index = ignore_index
        self.sample_rate = sample_rate
        self.frame_height = output_frame_size[1]
        self.frame_width = output_frame_size[0]
        self.need_label = need_label
        self.need_split = need_split
    
    def init_scores(self):
        self.imgs_list = []
        self.labels_list = []
        self.score_lsit = []
        self.init_flag = True

    def update(self, cam_images, labels, score, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        score = torch.transpose(score[-1], -1, -2)
        self.labels_list.append(labels)
        with torch.no_grad():
            pred = np.argmax(score.detach().cpu().numpy(), axis=-1)
            self.score_lsit.append(pred)
        for bs in range(cam_images.shape[0]):
            if len(self.imgs_list) < (bs + 1):
                self.imgs_list.append(CAMImageStreamWriter(self.frame_height, self.frame_width, need_label=self.need_label, need_split=self.need_split))
            self.imgs_list[bs].stream_write(cam_images[bs])

    def output(self):
        imags_list = []
        labels_list = []
        preds_list = []

        labels = np.concatenate(self.labels_list, axis=1)
        preds = np.concatenate(self.score_lsit, axis=1)

        for bs in range(len(self.imgs_list)):
            index = np.where(labels[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [labels[bs].shape[-1]]) // self.sample_rate
            self.imgs_list[bs].dump()
            imags_list.append({"writer":self.imgs_list[bs], "len":ignore_start})
            labels_list.append(labels[bs, ::self.sample_rate][:ignore_start])
            preds_list.append(preds[bs, ::self.sample_rate][:ignore_start])

        return imags_list, labels_list, preds_list