'''
Author: Thyssen Wen
Date: 2022-03-21 15:22:51
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-16 09:57:23
Description: runner script
FilePath     : /ETESVS/tasks/runner.py
'''
import torch
import time
from utils.logger import log_batch
import torch.distributed as dist

from utils.logger import get_logger
import numpy as np
import cv2 
try:
    from apex import amp
except:
    pass

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM) # sum-up as the all-reduce operation
    rt /= nprocs # NOTE this is necessary, since all_reduce here do not perform average 
    return rt

class Runner():
    def __init__(self,
                 logger,
                 video_batch_size,
                 Metric,
                 cfg,
                 model,
                 post_processing,
                 record_dict=None,
                 criterion=None,
                 optimizer=None,
                 use_amp=False,
                 nprocs=1,
                 local_rank=-1,
                 runner_mode='train'):
        self.optimizer = optimizer
        self.logger = logger
        self.video_batch_size = video_batch_size
        self.Metric = Metric
        self.record_dict = record_dict
        self.cfg = cfg
        self.model = model
        self.criterion = criterion
        self.post_processing = post_processing
        self.nprocs = nprocs
        self.local_rank = local_rank
        self.use_amp = use_amp

        # self.writer = get_logger(name="SVTAS", tensorboard=True)
        # self.step = 1
        # self.cnt = 1
        self.segs = 0 
        assert runner_mode in ['train', 'validation', 'test'], "Not support this runner mode: " + runner_mode
        self.runner_mode = runner_mode
    
    def _init_loss_dict(self):
        self.loss_dict = {}
        if self.record_dict is not None:
            for key, _ in self.record_dict.items():
                if key.endswith("loss"):
                    self.loss_dict[key] = 0.
    
    def _update_loss_dict(self, input_loss_dict):
        for key, _ in self.loss_dict.items():
            self.loss_dict[key] = self.loss_dict[key] + input_loss_dict[key].detach().clone()
    
    def _distribute_sync_loss_dict(self):
        for key, value in self.loss_dict.items():
            if key != "loss":
                self.loss_dict[key] = reduce_mean(value, self.nprocs)

    def _log_loss_dict(self):
        for key, value in self.loss_dict.items(): 
            self.record_dict[key].update(value.item(), self.video_batch_size)

    def epoch_init(self):
        # batch videos sampler
        self._init_loss_dict()
        self.action_seg_acc = 0.
        self.branch_seg_acc = 0.

        self.post_processing.init_flag = False
        self.current_step = 0
        self.current_step_vid_list = None

        if self.runner_mode in ['train']:
            self.model.train()
            self.b_tic = time.time()
            # reset recoder
            for _, recod in self.record_dict.items():
                recod.reset()

        elif self.runner_mode in ['validation', 'test']:
            self.model.eval()
            self.b_tic = time.time()
            # reset recoder
            for _, recod in self.record_dict.items():
                recod.reset()
        
        
    def batch_end_step(self, sliding_num, vid_list, step, epoch):
        if self.runner_mode in ['train']:
            # for name, param in self.model.named_parameters():
            #     self.writer.add_histogram(name, param.clone().cpu().data.numpy(), self.step)
            #     self.writer.add_histogram(name + '/grad', param.grad.clone().cpu().data.numpy(), self.step)
            # self.step = self.step + 1

            self.optimizer.step()
            self.optimizer.zero_grad()

        # clear memory buffer
        if self.nprocs > 1:
            self.model.module._clear_memory_buffer()
        else:
            self.model._clear_memory_buffer()

        # get pred result
        # pred_score_list, pred_cls_list, ground_truth_list,  = self.post_processing.output()

        def process_post_output(postprocessed, runner_mode, nprocs, current_step_vid_list):
            """
            Process post-processing outputs and optionally handle distributed gathering.

            Args:
                post_processing: An object with a method `output()` that returns
                                pred_score_list, pred_cls_list, ground_truth_list.
                runner_mode: Mode of the runner, either 'validation', 'test', or others.
                nprocs: Number of processes in distributed training.
                current_step_vid_list: Current step video ID list.

            Returns:
                tuple: (outputs, ground_truth_list, vid)
            """
            # Get the initial output from post-processing
            pred_score_list, pred_cls_list, ground_truth_list = postprocessed
            outputs = dict(
                predict=pred_cls_list,
                output_np=pred_score_list
            )
            vid = current_step_vid_list

            # Handle distributed gathering if in validation or test mode
            # if runner_mode in ['validation', 'test']:
            #     if nprocs > 1:
            #         # Prepare collect_dict for distributed gathering
            #         collect_dict = dict(
            #             predict=pred_cls_list,
            #             output_np=pred_score_list,
            #             ground_truth=ground_truth_list,
            #             vid=current_step_vid_list
            #         )
            #         gather_objects = [collect_dict for _ in range(nprocs)]  # Prepare objects to gather
            #         output_list = [None for _ in range(nprocs)]

            #         # Perform distributed all_gather_object
            #         dist.all_gather_object(output_list, gather_objects[dist.get_rank()])

            #         # Collect gathered data from all processes
            #         pred_cls_list_i = []
            #         pred_score_list_i = []
            #         ground_truth_list_i = []
            #         vid_i = []

            #         for output_dict in output_list:
            #             pred_cls_list_i += output_dict["predict"]
            #             pred_score_list_i += output_dict["output_np"]
            #             ground_truth_list_i += output_dict["ground_truth"]
            #             vid_i += output_dict["vid"]

            #         # Update outputs and ground_truth_list with gathered data
            #         outputs = dict(
            #             predict=pred_cls_list_i,
            #             output_np=pred_score_list_i
            #         )
            #         ground_truth_list = ground_truth_list_i
            #         vid = vid_i

            return outputs, ground_truth_list, vid

        out = self.post_processing.output()
        outputs, ground_truth_list, vid = process_post_output(out[0], self.runner_mode, self.nprocs, self.current_step_vid_list)
        if not vid:
            import pdb; pdb.set_trace()
        f1_action, acc_action = self.Metric.update(vid, ground_truth_list, outputs, action_dict_path='data/thal/mapping_tasks.txt')
        outputs, ground_truth_list, vid = process_post_output(out[1], self.runner_mode, self.nprocs, self.current_step_vid_list)
        if not vid:
            import pdb; pdb.set_trace()
        f1_branch, acc_branch = self.Metric.update(vid, ground_truth_list, outputs, action_dict_path='data/thal/mapping_branches.txt')


        self.current_step_vid_list = vid_list
        if len(self.current_step_vid_list) > 0:
            self.post_processing.init_scores(sliding_num, len(vid_list))
        

        if self.nprocs > 1:
            torch.distributed.barrier()
            self._distribute_sync_loss_dict()

        # logger
        if self.runner_mode in ['train']:
            self.record_dict['lr'].update(self.optimizer.state_dict()['param_groups'][0]['lr'], self.video_batch_size)
        
        self._log_loss_dict()
        self.record_dict['batch_time'].update(time.time() - self.b_tic)
        self.record_dict['F1Action@0.5'].update(f1_action, self.video_batch_size)
        self.record_dict['F1Branch@0.5'].update(f1_branch, self.video_batch_size)
        self.record_dict['ActionAcc'].update(acc_action, self.video_batch_size)
        self.record_dict['BranchAcc'].update(acc_branch, self.video_batch_size)
        self.record_dict['ActionSeg_Acc'].update(self.action_seg_acc, self.video_batch_size)
        self.record_dict['BranchSeg_Acc'].update(self.branch_seg_acc, self.video_batch_size)

        self._init_loss_dict()
        self.action_seg_acc = 0.
        self.branch_seg_acc = 0.

        if self.current_step % self.cfg.get("log_interval", 10) == 0:
            ips = "ips: {:.5f} instance/sec.".format(
                self.video_batch_size / self.record_dict["batch_time"].val)
            if self.runner_mode in ['train']:
                log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "train", ips, self.logger)
            elif self.runner_mode in ['validation']:
                log_batch(self.record_dict, self.current_step, epoch + 1, self.cfg.epochs, "validation", ips, self.logger)
            elif self.runner_mode in ['test']:
                log_batch(self.record_dict, self.current_step, 1, 1, "test", ips, self.logger)

        self.b_tic = time.time()

        self.current_step = step

    def postprocess_frame(self, frame_list, wh =  (1280, 720)):
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])

        original_frames = []
        for frame in frame_list:
            unnormalized_frame = frame * std[:, None, None] + mean[:, None, None]
            uint8_frame = (unnormalized_frame.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            original_frames.append(cv2.resize(uint8_frame, wh))
        
        return original_frames
    def _model_forward(self, data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                input_data[key] = value.cuda()
        
        outputs = self.model(input_data)
        loss_dict = self.criterion(outputs, input_data)
        loss = loss_dict["loss"]
        if self.runner_mode in ['train']:
            if self.use_amp is True:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
        if not torch.is_tensor(outputs) and not isinstance(outputs, dict):
            outputs = outputs[-1]
            
        return outputs, loss_dict

    def run_one_clip(self, data_dict):
        vid_list = data_dict['vid_list']
        sliding_num = data_dict['sliding_num']
        idx = data_dict['current_sliding_cnt']
        labels = data_dict['labels']
        branch_labels = data_dict['branch_labels']
        # train segment
        # self.segs += 1 
        # mean = torch.tensor([0.485, 0.456, 0.406])
        # std = torch.tensor([0.229, 0.224, 0.225])
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(f'input_segments/{self.segs}.mp4', fourcc, 30, (224, 224))
        # for frame in data_dict['imgs'][0]: 
        #     unnorm = frame
        #     norm = unnorm * std[:, None, None] + mean[:, None, None]
        #     norm = (norm.numpy()*255).astype(np.uint8).transpose(1,2,0)[:,:,::-1]
        #     out.write(norm)

        # out.release()
        if self.nprocs > 1 and idx < sliding_num - 1 and self.use_amp is False:
            with self.model.no_sync():
                # multi-gpus
                score, loss_dict = self._model_forward(data_dict)
        else:
            # single gpu
            score, loss_dict = self._model_forward(data_dict)

        # score = score.unsqueeze(0)
        # score = torch.nn.functional.interpolate(
        #     input=score,
        #     scale_factor=[1, 4],
        #     mode="nearest")
        with torch.no_grad():
            if self.post_processing.init_flag is not True:
                self.post_processing.init_scores(sliding_num, len(vid_list))
                self.current_step_vid_list = vid_list
            accs = [acc / sliding_num for acc in list(self.post_processing.update(score['action_score'],score['branch_score'], labels, branch_labels, idx))]
            self.action_seg_acc += accs[0]
            self.branch_seg_acc += accs[1]


            # logger loss
            self._update_loss_dict(loss_dict)

    def run_one_iter(self, data, r_tic=None, epoch=None):
        # videos sliding stream train
        self.record_dict['reader_time'].update(time.time() - r_tic)

        for sliding_seg in data:
            
            step = sliding_seg['step']
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']
            # wheather next step

            if self.current_step != step or (len(vid_list) <= 0 and step == 1):
                self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)

            if idx >= 0: 
                self.run_one_clip(sliding_seg)
    
    def run_one_batch(self, data, r_tic=None, epoch=None):
        # videos batch train
        self.record_dict['reader_time'].update(time.time() - r_tic)

        for sliding_seg in data:
            step = self.current_step
            vid_list = sliding_seg['vid_list']
            sliding_num = sliding_seg['sliding_num']
            idx = sliding_seg['current_sliding_cnt']

            # run one batch
            self.run_one_clip(sliding_seg)
            self.batch_end_step(sliding_num=sliding_num, vid_list=vid_list, step=step, epoch=epoch)
            self.current_step = self.current_step + 1