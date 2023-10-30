'''
Author       : Thyssen Wen
Date         : 2023-10-30 15:28:23
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 21:29:02
Description  : file content
FilePath     : \ETESVS\svtas\serving\client\visualizer\opencv_visualizer.py
'''
import os
import copy
import queue
import numpy as np
from typing import Dict, List
from threading import Thread
from .base import BaseClientViusalizer
from svtas.utils import AbstractBuildFactory, is_opencv_available
from svtas.utils.misc import make_palette, label_arr2img, draw_action_label

if is_opencv_available():
    import cv2

@AbstractBuildFactory.register('serving_client_visualizer')
class OpencvViusalizer(BaseClientViusalizer):
    END_FLAG = False

    def __init__(self,
                 label_path: str,
                 clip_seg_num: int,
                 sample_rate: int,
                 memory_factor: int = 3,
                 vis_size: List = [1280, 720],
                 fps: int = 30,
                 save_path: str = None) -> None:
        super().__init__()
        self.save_path = save_path
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
        self.memory_factor = memory_factor
        # get video width
        self.frame_width = vis_size[0]
        # get video height
        self.frame_height = vis_size[1]
        # get video fps
        fps = fps

        self.video_writer = None
        if save_path is not None:
            output_path = os.path.join(save_path, "inference.mp4")
            # write video setting
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(output_path, fourcc, fps, (self.frame_width, self.frame_height))

        # labels
        file_ptr = open(label_path, 'r')
        actions = file_ptr.read().split('\n')[:-1]
        file_ptr.close()
        self.actions_dict = dict()
        for a in actions:
            self.actions_dict[int(a.split()[0])] = a.split()[1]
        self.palette = make_palette(len(self.actions_dict))

        # buffer
        self.label_queue = queue.Queue(maxsize=self.clip_seg_num * self.sample_rate * self.memory_factor)
        self.show_queue = queue.Queue()
        self.thread_pool = []

    def init(self):
        return super().init()

    def update_show_data(self, show_data_dict: Dict):
        imgs = show_data_dict['raw_imgs']
        for i, img in enumerate(imgs):
            out_frame = copy.deepcopy(img)
            label_id = show_data_dict['outputs']['predict'][0][i]
            # add infer info
            cv2.putText(out_frame, "Prediction: " + self.actions_dict[label_id], (0, self.frame_height - 60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
            # cv2.putText(out_frame, "FPS: " + "{:.2f}".format(chunk_fps), (frame_width - 150, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
            if self.label_queue.full():
                self.label_queue.get()
            self.label_queue.put([label_id])
            label_img = label_arr2img(self.label_queue, self.palette).convert('RGB')
            past_width = int((label_img.size[0] / (self.clip_seg_num * self.sample_rate * self.memory_factor)) * (self.frame_width - 30))
            label_img = cv2.cvtColor(np.asarray(label_img),cv2.COLOR_RGB2BGR)
            label_img = cv2.resize(label_img, (past_width, 20))
            out_frame[(self.frame_height - 30):(self.frame_height - 10), 10:(10 + past_width), :] = label_img
            out_frame = cv2.rectangle(out_frame, (10 + past_width, self.frame_height - 10), (20 + past_width, self.frame_height - 30), (255, 255, 255), thickness=-1)
            cv2.putText(out_frame, "Current Frame", (max(past_width - 120, 0), self.frame_height - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            data = list(copy.deepcopy(self.label_queue.queue))
            array = np.array(data).transpose()
            label = list(set(array[0, :].tolist()))
            out_frame = draw_action_label(out_frame, self.palette, self.actions_dict, label)
            self.show_queue.put(out_frame)

            if self.video_writer:
                self.video_writer.write(out_frame)
    
    @staticmethod
    def set_end_flag():
        OpencvViusalizer.END_FLAG = True

    @staticmethod
    def cv2_windows_show(show_queue: queue.Queue):
        while True:
            if not show_queue.empty() and not OpencvViusalizer.END_FLAG:
                img = show_queue.get()
                cv2.imshow("tas_client", img)
                if cv2.waitKey(10) == ord('q'):
                    break
            elif OpencvViusalizer.END_FLAG:
                return
                
    def show(self):
        self.thread_pool.append(Thread(target=self.cv2_windows_show, args=(self.show_queue,)))

        for t in self.thread_pool:
            t.start()

    def shutdown(self):
        self.set_end_flag()
        if self.video_writer:
            self.video_writer.release()
        for t in self.thread_pool:
            t.join()
        cv2.destroyAllWindows()
        self.logger.log("Shutdown Viusalizer successfully!")