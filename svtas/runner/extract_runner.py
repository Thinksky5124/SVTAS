'''
Author       : Thyssen Wen
Date         : 2022-10-27 19:01:22
LastEditors  : Thyssen Wen
LastEditTime : 2022-10-27 19:01:34
Description  : Extract Runner Class
FilePath     : /SVTAS/svtas/runner/extract_runner.py
'''
import numpy as np
import os
import torch

class ExtractRunner():
    def __init__(self,
                 logger,
                 model,
                 post_processing,
                 feature_out_path,
                 logger_interval=10):
        self.model = model
        self.logger = logger
        self.post_processing = post_processing
        self.feature_out_path = feature_out_path
        self.logger_interval = logger_interval
    
    def epoch_init(self):
        # batch videos sampler
        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None
        self.model.eval()
    
    @torch.no_grad()
    def batch_end_step(self, sliding_num, vid_list, step):

        # get extract feature
        extract_feature_list = self.post_processing.output()
        
        # save feature file
        current_vid_list = self.current_step_vid_list
        for extract_feature, vid in zip(extract_feature_list, current_vid_list):
            feature_save_path = os.path.join(self.feature_out_path, vid + ".npy")
            np.save(feature_save_path, extract_feature)

        self.logger.info("Step: " + str(step) + ", finish ectracting video: "+ ",".join(current_vid_list))
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