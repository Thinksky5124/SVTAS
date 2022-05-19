'''
Author       : Thyssen Wen
Date         : 2022-05-17 16:58:53
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-18 14:57:48
Description  : Extract video feature script
FilePath     : /ETESVS/tools/extract_features.py
'''
import os
import sys
from tkinter.messagebox import NO
path = os.path.join(os.getcwd())
sys.path.append(path)
import torch
import numpy as np
import model.builder as model_builder
import loader.builder as dataset_builder
import argparse
from utils.config import parse_config
from utils.logger import get_logger, setup_logger
from mmcv.runner import load_state_dict

class ExtractRunner():
    def __init__(self,
                 logger,
                 model,
                 post_processing,
                 feature_out_path):
        self.model = model
        self.logger = logger
        self.post_processing = post_processing
        self.feature_out_path = feature_out_path
    
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
            
        return outputs[-1]
    
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
            self.post_processing.update(score, labels, idx)

    @torch.no_grad()
    def run_one_iter(self, data):
        # videos sliding stream train
        for sliding_seg in data:
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step
            if self.current_step != step and not (self.current_step == 0 and len(vid_list) <= 0):
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step)

            if idx >= 0: 
                self.run_one_clip(sliding_seg)

@torch.no_grad()
def extractor(cfg, outpath):
    out_path = os.path.join(outpath, "features")
    isExists = os.path.exists(out_path)
    if not isExists:
        os.makedirs(out_path)
        print(out_path + ' created successful')
    logger = get_logger("ETESVS")
    
    # construct model
    model = model_builder.build_model(cfg.MODEL).cuda()

    pretrain_path = cfg.get('PRETRAINED', None)
    if pretrain_path is not None:
        checkpoint = torch.load(pretrain_path)
        state_dicts = checkpoint["model_state_dict"]
        load_state_dict(model, state_dicts, logger=logger)
        # model.load_state_dict(state_dicts)

    # construct dataloader
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = dataset_builder.build_pipline(cfg.COLLATE)
    Pipeline = dataset_builder.build_pipline(cfg.PIPELINE)
    dataset_config = cfg.DATASET.config
    dataset_config['pipeline'] = Pipeline
    dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    dataset_config['video_batch_size'] = video_batch_size
    dataloader = torch.utils.data.DataLoader(
        dataset_builder.build_dataset(dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)
    
    post_processing = model_builder.build_post_precessing(cfg.POSTPRECESSING)

    runner = ExtractRunner(logger=logger, model=model, post_processing=post_processing, feature_out_path=out_path)

    runner.epoch_init()
    for i, data in enumerate(dataloader):
        runner.run_one_iter(data=data)
    
    logger.info("Finish all extracting!")

def parse_args():
    parser = argparse.ArgumentParser("ETESVS extract video feature script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('-o',
                        '--out_path',
                        type=str,
                        help='extract flow file out path')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args
        
def main():
    args = parse_args()
    setup_logger(f"./output/etract_feature", name="ETESVS", level="INFO", tensorboard=False)
    cfg = parse_config(args.config)
    extractor(cfg, args.out_path)

if __name__ == '__main__':
    main()