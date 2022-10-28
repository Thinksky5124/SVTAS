'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:01:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-28 10:04:13
Description  : Extract Runner Class
FilePath     : /SVTAS/svtas/runner/extract_runner.py
'''
from abc import abstractclassmethod
import cv2
import numpy as np
import os
import torch

class ExtractRunner():
    def __init__(self,
                 logger,
                 model,
                 post_processing,
                 out_path,
                 logger_interval=10):
        self.model = model
        self.logger = logger
        self.post_processing = post_processing
        self.out_path = out_path
        self.logger_interval = logger_interval
    
    def epoch_init(self):
        # batch videos sampler
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.model.eval()
        self.init_file_dir()
    
    @abstractclassmethod
    def init_file_dir(self):
        pass
    
    @abstractclassmethod
    def duil_will_end_extract(self, extract_output, current_vid_list):
        raise NotImplementedError()
    
    @torch.no_grad()
    def batch_end_step(self, sliding_num, vid_list, step):
        self.model._clear_memory_buffer()
        # get extract feature
        extract_output = self.post_processing.output()
        
        # save feature file
        self.duil_will_end_extract(extract_output, self.current_step_vid_list)

        self.logger.info("Step: " + str(step) + ", finish ectracting video: "+ ",".join(self.current_step_vid_list))
        self.current_step_vid_list = vid_list
        
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))

        self.current_step = step
    
    @torch.no_grad()
    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                input_data[key] = value.cuda()

        outputs = self.model(input_data)
        
        if not torch.is_tensor(outputs):
            outputs = outputs[-1]
            
        return outputs
    
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
            self.post_processing.update(score, labels, idx)
        
        if idx % self.logger_interval == 0:
            self.logger.info("Current process idx: " + str(idx) + " | total: " + str(sliding_num))

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

class ExtractFeatureRunner(ExtractRunner):

    def init_file_dir(self):
        pass

    def duil_will_end_extract(self, extract_output, current_vid_list):
        for extract_feature, vid in zip(extract_output, current_vid_list):
            feature_save_path = os.path.join(self.out_path, vid + ".npy")
            np.save(feature_save_path, extract_feature)

class ExtractOpticalFlowRunner(ExtractRunner):

    def init_file_dir(self):
        self.flow_out_path = os.path.join(self.out_path, "flow")
        isExists = os.path.exists(self.flow_out_path)
        if not isExists:
            os.makedirs(self.flow_out_path)
            print(self.flow_out_path + ' created successful')

        if self.post_processing.need_visualize:
            self.video_out_path = os.path.join(self.out_path, "flow_video")
            isExists = os.path.exists(self.video_out_path)
            if not isExists:
                os.makedirs(self.video_out_path)
                print(self.video_out_path + ' created successful')

    def duil_will_end_extract(self, extract_output, current_vid_list):
        if len(extract_output) > 1:
            flow_imgs_list, flow_visual_imgs_list, fps = extract_output
            for extract_flow, flow_visual_imgs, vid in zip(flow_imgs_list, flow_visual_imgs_list, current_vid_list):
                optical_flow_save_path = os.path.join(self.flow_out_path, vid + ".npy")
                np.save(optical_flow_save_path, extract_flow)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                flow_video_path = os.path.join(self.video_out_path, vid + '.mp4')
                videoWrite = cv2.VideoWriter(flow_video_path, fourcc, fps, (flow_visual_imgs.shape[-2], flow_visual_imgs.shape[-3]))

                for flow_img in flow_visual_imgs:
                    videoWrite.write(flow_img)
                
                videoWrite.release()
                
        else:
            for extract_flow, vid in zip(extract_output, current_vid_list):
                optical_flow_save_path = os.path.join(self.flow_out_path, vid + ".npy")
                np.save(optical_flow_save_path, extract_flow)