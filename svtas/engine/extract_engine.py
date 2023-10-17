'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:01:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-16 23:33:30
Description  : Extract Engine Class
FilePath     : /SVTAS/svtas/engine/extract_engine.py
'''
from typing import Any, Dict
import cv2
import numpy as np
import os
import torch

from svtas.loader.dataloader import BaseDataloader
from .standalone_engine import StandaloneEngine
from svtas.utils.logger import AverageMeter
from svtas.model_pipline import FakeModelPipline
from svtas.utils import AbstractBuildFactory
from svtas.utils.loss_landspace import (create_random_directions, plot_landspace_1D_loss_err,
                            calulate_loss_landscape, plot_landspace_2D_loss_err, caculate_trajectory,
                            plot_contour_trajectory)

@AbstractBuildFactory.register('engine')
class BaseExtractEngine(StandaloneEngine):
    def __init__(self,
                 model_name: str,
                 post_processing: Dict,
                 out_path: str,
                 logger_dict: Dict,
                 record: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 metric: Dict = {}) -> None:
        super().__init__(model_name, FakeModelPipline(post_processing), logger_dict, record,
                         metric, iter_method, checkpointor, "extract")
        self.out_path = out_path
        isExists = os.path.exists(self.out_path)
        if not isExists:
            os.makedirs(self.out_path)
            print(self.out_path + ' created successful')
    
    def init_engine(self, dataloader: BaseDataloader = None):
        self.init_file_dir()
        self.iter_method.register_every_batch_end_hook(self.duil_will_end_extract)
        if hasattr(self.iter_method, "register_every_iter_end_hook"):
            self.iter_method.register_every_iter_end_hook(self.duil_will_iter_end_extract)
        return super().init_engine(dataloader)
    
    @classmethod
    def init_file_dir(self):
        pass
    
    @classmethod
    def duil_will_iter_end_extract(self, extract_output, current_vid_list):
        pass
    
    @classmethod
    def duil_will_end_extract(self, extract_output, current_vid_list):
        pass

@AbstractBuildFactory.register('engine')
class ExtractModelEngine(StandaloneEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 out_path: str,
                 logger_dict: Dict,
                 record: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 metric: Dict = {},
                 running_mode='extract') -> None:
        super().__init__(model_name, model_pipline, logger_dict, record,
                         metric, iter_method, checkpointor, running_mode)
        self.out_path = out_path
        isExists = os.path.exists(self.out_path)
        if not isExists:
            os.makedirs(self.out_path)
            print(self.out_path + ' created successful')
        
    def init_engine(self, dataloader: BaseDataloader = None):
        self.init_file_dir()
        self.iter_method.register_every_batch_end_hook(self.duil_will_end_extract)
        if hasattr(self.iter_method, "register_every_iter_end_hook"):
            self.iter_method.register_every_iter_end_hook(self.duil_will_iter_end_extract)
        return super().init_engine(dataloader)
    
    @classmethod
    def init_file_dir(self):
        pass
    
    @classmethod
    def duil_will_iter_end_extract(self, extract_output, current_vid_list):
        pass
    
    @classmethod
    def duil_will_end_extract(self, extract_output, current_vid_list):
        pass

@AbstractBuildFactory.register('engine')
class LossLandSpaceEngine(ExtractModelEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 out_path: str,
                 weight_path: str,
                 logger_dict: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 xmin=-0.5,
                 xmax=0.5,
                 xnum=51,
                 ymin=-0.5,
                 ymax=0.5,
                 ynum=51,
                 vmin = 0,
                 vmax = 10,
                 vlevel = 0.05,
                 plot_1D=False,
                 plot_optimization_path=False,
                 record: Dict = {},
                 running_mode='extract') -> None:
        if 'addition_record' not in record:
            record['addition_record'] += [dict(name='loss_sample', fmt='.5f')]
            record['accumulate_type']['loss_sample'] = 'avg'
        super().__init__(model_name, model_pipline, out_path, logger_dict,
                         record, iter_method, checkpointor, metric, running_mode)
        self.xmin, self.xmax, self.xnum = xmin, xmax, xnum
        self.ymin, self.ymin, self.ynum = ymin, ymax, ynum
        self.vmin, self.vmax, self.vlevel = vmin, vmax, vlevel
        self.plot_1D = plot_1D
        self.plot_optimization_path = plot_optimization_path
        self.weight_path = weight_path
    
    def set_all_dataloader(self, train_dataloader: BaseDataloader, test_dataloader: BaseDataloader):
        self.test_dataloader = test_dataloader
        self.train_dataloader = train_dataloader

    def duil_will_iter_end_losslandspace(self, extract_output, current_step_vid_list):
        pass
    
    def duil_will_end_extract(self, extract_output, current_vid_list):
        pass
    
    def run(self):
        super().run()
        logger = self.logger_dict['SVTAS'].logger
        rand_directions = create_random_directions(self.model_pipline.model, plot_1D=self.plot_1D)
        if not self.plot_1D:
            calulate_loss_landscape(self.model_pipline.model, rand_directions, self.out_path, logger, self, self.test_dataloader, False, self.iter_method.criterion_metric_name, key='test',
                                    xmin=self.xmin, xmax=self.xmax, xnum=self.xnum, ymin=self.ymin, ymax=self.ymax, ynum=self.ynum)
            plot_landspace_2D_loss_err(self.out_path, logger, self.iter_method.criterion_metric_name, vmin = self.vmin, vmax = self.vmax, vlevel = self.vlevel, surf_name = "test_loss")
        else:
            calulate_loss_landscape(self.model_pipline.model, rand_directions, self.out_path, logger, self, self.test_dataloader, True, self.iter_method.criterion_metric_name, key='test',
                                xmin=self.xmin, xmax=self.xmax, xnum=self.xnum, ymin=self.ymin, ymax=self.ymax, ynum=self.ynum)
            calulate_loss_landscape(self.model_pipline.model, rand_directions, self.out_path, logger, self, self.train_dataloader, True, self.iter_method.criterion_metric_name, key='train',
                                xmin=self.xmin, xmax=self.xmax, xnum=self.xnum, ymin=self.ymin, ymax=self.ymax, ynum=self.ynum)
            plot_landspace_1D_loss_err(self.out_path, logger, self.iter_method.criterion_metric_name, xmin=self.xmin, xmax=self.xmax, loss_max=5, log=False, show=False)
        if self.plot_optimization_path:
            caculate_trajectory(self.model_name, self.weight_path, self.model_pipline.model, self.out_path, logger, dir_type='weights', ignore_bias=False)
            plot_contour_trajectory(self.out_path, logger, surf_name='loss_vals', vmin=self.vmin, vmax=self.vmax, vlevel=self.vlevel, show=False)
    
@AbstractBuildFactory.register('engine')
class ExtractFeatureEngine(ExtractModelEngine):

    def init_file_dir(self):
        pass

    def duil_will_end_extract(self, extract_output, current_vid_list):
        for extract_feature, vid in zip(extract_output, current_vid_list):
            feature_save_path = os.path.join(self.out_path, vid + ".npy")
            stream_writer, v_len = extract_feature["writer"], extract_feature["len"]
            stream_writer.save(feature_save_path, v_len)

@AbstractBuildFactory.register('engine')
class ExtractOpticalFlowEngine(ExtractModelEngine):

    def init_file_dir(self):
        self.flow_out_path = os.path.join(self.out_path, "flow")
        isExists = os.path.exists(self.flow_out_path)
        if not isExists:
            os.makedirs(self.flow_out_path)
            for key, logger in self.logger_dict.items():
                logger.log(self.flow_out_path + ' created successful')

        if self.model_pipline.post_processing.need_visualize:
            self.video_out_path = os.path.join(self.out_path, "flow_videos")
            isExists = os.path.exists(self.video_out_path)
            if not isExists:
                os.makedirs(self.video_out_path)
                for key, logger in self.logger_dict.items():
                    logger.log(self.video_out_path + ' created successful')

    def duil_will_end_extract(self, extract_output, current_vid_list):
        if len(extract_output) > 1:
            flow_imgs_list, flow_visual_imgs_list = extract_output
            for extract_flow, flow_visual_imgs, vid in zip(flow_imgs_list, flow_visual_imgs_list, current_vid_list):
                optical_flow_save_path = os.path.join(self.flow_out_path, vid + ".mp4")
                stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                stream_writer.save(optical_flow_save_path, v_len)

                video_out_path = os.path.join(self.video_out_path, vid + ".mp4")
                stream_writer, v_len = flow_visual_imgs["writer"], extract_flow["len"]
                stream_writer.save(video_out_path, v_len)
                
        else:
            for extract_flow, vid in zip(extract_output, current_vid_list):
                optical_flow_save_path = os.path.join(self.flow_out_path, vid + ".mp4")
                stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                stream_writer.save(optical_flow_save_path, v_len)

@AbstractBuildFactory.register('engine')
class ExtractMVResEngine(BaseExtractEngine):
    def __init__(self,
                 model_name: str,
                 post_processing: Dict,
                 out_path: str,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 res_extract: bool = True,
                 running_mode: str = 'extract') -> None:
        super().__init__(model_name, post_processing, out_path, logger_dict,
                         record, metric, iter_method, checkpointor, running_mode)
        self.res_extract = res_extract
    
    def init_file_dir(self):
        if self.res_extract:
            res_out_path = os.path.join(self.out_path, "res_videos")
            isExists = os.path.exists(res_out_path)
            if not isExists:
                os.makedirs(res_out_path)
                print(res_out_path + ' created successful')
            self.res_out_path = res_out_path
        mvs_outpath = os.path.join(self.out_path, "mvs_videos")
        if self.model_pipline.post_processing.need_visualize:
            mvs_vis_outpath = os.path.join(self.out_path, "mvs_vis_videos")
            isExists = os.path.exists(mvs_vis_outpath)
            if not isExists:
                os.makedirs(mvs_vis_outpath)
                print(mvs_outpath + ' created successful')
            self.mvs_vis_outpath = mvs_vis_outpath

        isExists = os.path.exists(mvs_outpath)
        if not isExists:
            os.makedirs(mvs_outpath)
            print(mvs_outpath + ' created successful')
        self.mvs_outpath = mvs_outpath
    
    def duil_will_iter_end_extract(self, extract_output, current_vid_list):
        pass
    
    def duil_will_end_extract(self, extract_output, current_vid_list):
        if len(extract_output) == 1:
            flow_imgs_list = extract_output[0]
            for extract_flow, vid in zip(flow_imgs_list, current_vid_list):
                mvs_outpath = os.path.join(self.mvs_outpath, vid + ".mp4")
                stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                stream_writer.save(mvs_outpath, v_len)

        elif len(extract_output) == 2:
            if self.model_pipline.post_processing.need_visualize:
                flow_imgs_list, flow_visual_imgs_list = extract_output
                for extract_flow, extract_flow_vis, vid in zip(flow_imgs_list, flow_visual_imgs_list, current_vid_list):
                    mvs_outpath = os.path.join(self.mvs_outpath, vid + ".mp4")
                    stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                    stream_writer.save(mvs_outpath, v_len)

                    mvs_vis_outpath = os.path.join(self.mvs_vis_outpath, vid + ".mp4")
                    stream_writer, v_len = extract_flow_vis["writer"], extract_flow_vis["len"]
                    stream_writer.save(mvs_vis_outpath, v_len)
            else:
                flow_imgs_list, res_imgs_list = extract_output
                for extract_flow, extract_res, vid in zip(flow_imgs_list, res_imgs_list, current_vid_list):
                    mvs_outpath = os.path.join(self.mvs_outpath, vid + ".mp4")
                    stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                    stream_writer.save(mvs_outpath, v_len)

                    res_out_path = os.path.join(self.res_out_path, vid + ".mp4")
                    stream_writer, v_len = extract_res["writer"], extract_res["len"]
                    stream_writer.save(res_out_path, v_len)
        elif len(extract_output) == 3:
            flow_imgs_list, res_imgs_list, flow_visual_imgs_list = extract_output
            for extract_flow, extract_res, extract_flow_vis, vid in zip(flow_imgs_list, res_imgs_list, flow_visual_imgs_list, current_vid_list):
                mvs_outpath = os.path.join(self.mvs_outpath, vid + ".mp4")
                stream_writer, v_len = extract_flow["writer"], extract_flow["len"]
                stream_writer.save(mvs_outpath, v_len)

                mvs_vis_outpath = os.path.join(self.mvs_vis_outpath, vid + ".mp4")
                stream_writer, v_len = extract_flow_vis["writer"], extract_flow_vis["len"]
                stream_writer.save(mvs_vis_outpath, v_len)

                res_out_path = os.path.join(self.res_out_path, vid + ".mp4")
                stream_writer, v_len = extract_res["writer"], extract_res["len"]
                stream_writer.save(res_out_path, v_len)