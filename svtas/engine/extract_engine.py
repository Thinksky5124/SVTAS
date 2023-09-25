'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:01:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-09-25 14:23:44
Description  : Extract Engine Class
FilePath     : /SVTAS/svtas/engine/extract_engine.py
'''
import cv2
import numpy as np
import os
import torch
from .base_engine import BaseEngine
from svtas.utils.recorder import AverageMeter

from svtas.utils import AbstractBuildFactory

@AbstractBuildFactory.register('engine')
class ExtractEngine(BaseEngine):
    def __init__(self,
                 logger,
                 post_processing,
                 out_path,
                 logger_interval=10):
        self.logger = logger
        self.post_processing = post_processing
        self.out_path = out_path
        self.logger_interval = logger_interval
    
    def epoch_init(self):
        # batch videos sampler
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.init_file_dir()

        # self.cnt = 0

    @classmethod
    def init_file_dir(self):
        pass
    
    @classmethod
    def duil_will_iter_end_extract(self, extract_output, current_vid_list):
        pass
    
    @classmethod
    def duil_will_end_extract(self, extract_output, current_vid_list):
        pass
    
    @torch.no_grad()
    def batch_end_step(self, sliding_num, vid_list, step):
        # get extract feature
        extract_output = self.post_processing.output()
        
        # save feature file
        self.duil_will_end_extract(extract_output, self.current_step_vid_list)

        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

        self.logger.info("Step: " + str(step) + ", finish ectracting video: "+ ",".join(self.current_step_vid_list))
        self.current_step_vid_list = vid_list

        self.current_step = step
    
    @torch.no_grad()
    def _model_forward(self, data_dict):
        return data_dict
    
    @torch.no_grad()
    def run_one_clip(self, data_dict):
        vid_list = data_dict['vid_list']
        sliding_num = data_dict['sliding_num']
        idx = data_dict['current_sliding_cnt']
        labels = data_dict['labels']
        # train segment
        score = self._model_forward(data_dict)
            
        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
                self.logger.info("Current process video: " + ",".join(vid_list))
            extract_output = self.post_processing.update(score, labels, idx)
        
            # save feature file
            self.duil_will_iter_end_extract(extract_output, self.current_step_vid_list)
        
        if idx % self.logger_interval == 0:
            self.logger.info("Current process idx: " + str(idx) + " | total: " + str(sliding_num))
        
        # np.save(f"output/test/raw_data_{self.cnt}.npy", data_dict["imgs"].detach().clone().numpy())
        # self.cnt = self.cnt+1

    @torch.no_grad()
    def run_one_iter(self, data):
        # videos sliding stream train
        for sliding_seg in data:
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step
            if self.current_step != step or (len(vid_list) <= 0 and step == 1):
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step)

            if idx >= 0: 
                self.run_one_clip(sliding_seg)

@AbstractBuildFactory.register('engine')
class ExtractModelEngine(ExtractEngine):
    def __init__(self,
                 logger,
                 model,
                 post_processing,
                 out_path,
                 logger_interval=10):
        super().__init__(
            logger = logger,
            post_processing = post_processing,
            out_path = out_path,
            logger_interval = logger_interval
        )
        self.model = model
        
    
    def epoch_init(self):
        super().epoch_init()
        self.model.eval()
    
    @torch.no_grad()
    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                input_data[key] = value.cuda()

        outputs = self.model(input_data)
        
        score = outputs['output']
            
        return score

@AbstractBuildFactory.register('engine')
class LossLandSpaceEngine(ExtractModelEngine):
    def __init__(self,
                 logger,
                 model,
                 post_processing,
                 Metric,
                 out_path,
                 criterion,
                 need_grad_accumulate=True,
                 logger_interval=10):
        super().__init__(logger, model, post_processing, out_path, logger_interval)
        self.criterion = criterion
        self.Metric = Metric
        self.need_grad_accumulate = need_grad_accumulate
        self.record_dict = {'loss': AverageMeter('loss'), 'loss_sample': AverageMeter('loss_sample')}

    def epoch_init(self):
        super().epoch_init()
        self.model.eval()
        self.record_dict['loss'].reset()
        self.record_dict['loss_sample'].reset()

    @torch.no_grad()
    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                input_data[key] = value.cuda()
        if not self.need_grad_accumulate:
            input_data['precise_sliding_num'] = torch.ones_like(input_data['precise_sliding_num'])

        outputs = self.model(input_data)
        loss_dict = self.criterion(outputs, input_data)
        
        score = outputs['output']
        loss = loss_dict["loss"]
            
        return score, loss

    def duil_will_iter_end_losslandspace(self, loss, current_step_vid_list):
        self.record_dict['loss_sample'].update(loss)
    
    def duil_will_end_extract(self, extract_output, current_vid_list):
        pred_score_list, pred_cls_list, ground_truth_list = extract_output
        outputs = dict(predict=pred_cls_list,
                        output_np=pred_score_list)
        for k, v in self.Metric.items():
            acc = v.update(current_vid_list, ground_truth_list, outputs)

        if not self.need_grad_accumulate:
            self.record_dict['loss'].update(self.record_dict['loss_sample'].get_mean)
        else:
            self.record_dict['loss'].update(self.record_dict['loss_sample'].get_sum)
        self.record_dict['loss_sample'].reset()

    @torch.no_grad()
    def run_one_clip(self, data_dict):
        vid_list = data_dict['vid_list']
        sliding_num = data_dict['sliding_num']
        idx = data_dict['current_sliding_cnt']
        labels = data_dict['labels']
        # train segment
        score, loss = self._model_forward(data_dict)
            
        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
                self.logger.info("Current process video: " + ",".join(vid_list))
            extract_output = self.post_processing.update(score, labels, idx)
        
            # save feature file
            self.duil_will_iter_end_losslandspace(loss, self.current_step_vid_list)
        
        if idx % self.logger_interval == 0:
            self.logger.info("Current process idx: " + str(idx) + " | total: " + str(sliding_num))

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
            print(self.flow_out_path + ' created successful')

        if self.post_processing.need_visualize:
            self.video_out_path = os.path.join(self.out_path, "flow_videos")
            isExists = os.path.exists(self.video_out_path)
            if not isExists:
                os.makedirs(self.video_out_path)
                print(self.video_out_path + ' created successful')

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
class ExtractMVResEngine(ExtractEngine):
    def __init__(self,
                 logger,
                 post_processing,
                 out_path,
                 res_extract=True,
                 logger_interval=10):
        self.logger = logger
        self.post_processing = post_processing
        self.out_path = out_path
        self.logger_interval = logger_interval
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
        if self.post_processing.need_visualize:
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
            if self.post_processing.need_visualize:
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