'''
Author       : Thyssen Wen
Date         : 2023-10-11 14:36:20
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 14:54:15
Description  : file content
FilePath     : /SVTAS/svtas/engine/profile_engine.py
'''
import os
from typing import Dict, List
import torch
import numpy as np

from svtas.utils.logger import get_root_logger_instance
from svtas.utils import is_mmcv_available, is_fvcore_available
if is_mmcv_available():
    from mmcv.cnn.utils.flops_counter import get_model_complexity_info

if is_fvcore_available():
    from fvcore.nn import FlopCountAnalysis, flop_count_table
from svtas.utils.misc import clever_format
from svtas.loader.dataloader import BaseDataloader
from .standalone_engine import StandaloneEngine

class TorchStandaloneProfileEngine(StandaloneEngine):
    def __init__(self,
                 model_name: str,
                 model_pipline: Dict,
                 logger_dict: Dict,
                 record: Dict,
                 metric: Dict,
                 iter_method: Dict,
                 checkpointor: Dict,
                 clip_seg_num: int,
                 sample_rate: int,
                 image_size: List[int],
                 wait = 1,
                 warmup = 1,
                 active = 3,
                 repeat = 2,
                 record_shapes = True,
                 profile_memory = True,
                 with_stack = True,
                 running_mode='profile') -> None:
        super().__init__(model_name, model_pipline, logger_dict, record,
                         metric, iter_method, checkpointor, running_mode)
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.self.clip_seg_num = clip_seg_num
        self.self.sample_rate = sample_rate
        self.image_size = image_size
        
    def run(self):
        # model param flops caculate
        if self.model_pipline.model.__class__.__name__ not in ["FeatureSegmentation"]:
            image_size = self.image_size
            x_shape = [self.clip_seg_num, 3, image_size, image_size]
            mask_shape = [self.clip_seg_num * self.sample_rate]
            labels_shape = [self.clip_seg_num * self.sample_rate]
            input_shape = (x_shape, mask_shape, labels_shape)
            def input_constructor(input_shape, optimal_batch_size=1):
                x_shape, mask_shape, labels_shape = input_shape
                x = torch.randn([optimal_batch_size] + x_shape).to(self.model_pipline.device)
                mask = torch.randn([optimal_batch_size] + mask_shape).to(self.model_pipline.device)
                label = torch.ones([optimal_batch_size] + labels_shape).to(self.model_pipline.device)
                return dict(input_data=dict(imgs=x, masks=mask, labels=label))
            dummy_input = input_constructor(input_shape)
        else:
            x_shape = [self.clip_seg_num, 2048]
            mask_shape = [self.clip_seg_num * self.sample_rate]
            labels_shape = [self.clip_seg_num * self.sample_rate]
            input_shape = (x_shape, mask_shape, labels_shape)
            def input_constructor(input_shape, optimal_batch_size=1):
                x_shape, mask_shape, labels_shape = input_shape
                x = torch.randn([optimal_batch_size] + x_shape).to(self.model_pipline.device)
                mask = torch.randn([optimal_batch_size] + mask_shape).to(self.model_pipline.device)
                label = torch.ones([optimal_batch_size] + labels_shape).to(self.model_pipline.device)
                return dict(input_data=dict(feature=x, masks=mask, labels=label))
            dummy_input = input_constructor(input_shape)
        
        self.model_pipline.train()
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=self.wait, warmup=self.warmup, active=self.active, repeat=self.repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(os.path.join(self.checkpointor.save_path, self.model_name)),
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack
            ) as prof:
                for epoch in self.iter_method.run():
                    pass
        
        logger = get_root_logger_instance()
        self.model_pipline.eval()
        profile_flops_flag = False
        if is_mmcv_available():
            # mmcv caculate param and flops
            logger.info("="*20)
            logger.info('Use mmcv get_model_complexity_info function')
            flops_number, params_number = get_model_complexity_info(self.model_pipline.model, input_shape=input_shape, input_constructor=input_constructor, print_per_layer_stat=False, as_strings=False)
            flops_per_image_number = flops_number / self.clip_seg_num
            flops, params = clever_format([flops_number, params_number], "%.6f")
            flops_per_image, params = clever_format([flops_per_image_number, params_number], "%.6f")
            logger.info("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=self.clip_seg_num))
            logger.info("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops + ", Total params:" + params)
            logger.info(f"Computed strength is {flops_number/params_number} FLOPS/Byte.")
            logger.info("="*20)
            profile_flops_flag = True

        if is_fvcore_available():
            # fvcore caculate param and flops
            logger.info('Use fvcore FlopCountAnalysis function')
            inputs = (dummy_input['input_data'])
            flops = FlopCountAnalysis(self.model_pipline.model, inputs)
            logger.info("flop_count_table: \n" + flop_count_table(flops))
            flops_number = flops.total()
            flops_per_image_number = flops_number / self.clip_seg_num
            flops = clever_format([flops_number], "%.6f")
            flops_per_image = clever_format([flops_per_image_number], "%.6f")
            logger.info("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=self.clip_seg_num))
            logger.info("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops)
            logger.info("="*20)
            profile_flops_flag = True

        if not profile_flops_flag:
            print('You should install mmcv or fvcore for profile FLOPs!')

        # model fps caculate
        dummy_input = dummy_input['input_data']
        logger.info('Caculate model fps (single frame test times)')
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = self.repeat
        timings = np.zeros((repetitions, 1))

        #GPU-WARM-UP
        for _ in range(10):
            _ = self.model_pipline(dummy_input)

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.model_pipline(dummy_input)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        mean_fps = 1000. / mean_syn * self.clip_seg_num
        logger.info('Mean@1 {mean_syn:.3f}ms, Std@5 {std_syn:.3f}ms, FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
        logger.info('Model single forward test time(ms) {mean_syn:.3f}ms'.format(mean_syn=mean_syn))
        logger.info("="*20)

        # model latency time
        logger.info('Caculate model Throughput')
        repetitions=self.repeat
        total_time = 0
        # it should be modify by every model
        optimal_batch_size=1
        dummy_input = input_constructor(input_shape, optimal_batch_size=optimal_batch_size)['input_data']
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = self.model_pipline(dummy_input)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender) / 1000
                total_time += curr_time
        Throughput = (repetitions * optimal_batch_size) / total_time
        logger.info("Final Throughput: {Throughput:.2f} V/s, Measuring by batch_size: {Batch_size:d}".format(Throughput=Throughput, Batch_size=optimal_batch_size))
        logger.info("="*20)