'''
Author       : Thyssen Wen
Date         : 2022-10-31 19:02:43
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-08 15:43:12
Description  : file content
FilePath     : /SVTAS/svtas/engine/visual_engine.py
'''
import os
import math
from typing import Dict
import torch
from svtas.loader.dataloader import BaseDataloader

from svtas.utils.flow_vis import make_palette
from svtas.utils import AbstractBuildFactory
from .extract_engine import ExtractModelEngine

@AbstractBuildFactory.register('engine')
class VisualEngine(ExtractModelEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 out_path: str,
                 label_path: str,
                 logger_dict: Dict,
                 record: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 metric: Dict = {},) -> None:
        super().__init__(model_name, model_pipline, out_path, logger_dict, record,
                         iter_method, checkpointor, metric, 'visulaize')
        self.label_path = label_path
    
    def init_engine(self, dataloader: BaseDataloader = None):
        # load mapping label
        file_ptr = open(self.label_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        actions_dict = dict()
        for a in actions:
            actions_dict[int(a.split()[0])] = a.split()[1]
        self.palette = make_palette(len(actions_dict))
        self.actions_dict = actions_dict
        return super().init_engine(dataloader)
    
    def duil_will_end_extract(self, extract_output, current_vid_list):
        cam_imgs_list, labels_list, preds_list = extract_output
        for cam_imgs, vid, labels, preds in zip(cam_imgs_list, current_vid_list, labels_list, preds_list):
            cam_imgs_save_path = os.path.join(self.out_path, vid + ".mp4")
            stream_writer, v_len = cam_imgs["writer"], cam_imgs["len"]
            stream_writer.save(cam_imgs_save_path, v_len, labels, preds, self.actions_dict, self.palette)

    def duil_will_iter_end_extract(self, extract_output, current_vid_list):
        pass
    