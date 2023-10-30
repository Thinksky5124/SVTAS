'''
Author       : Thyssen Wen
Date         : 2023-10-30 17:04:24
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-30 21:01:16
Description  : file content
FilePath     : \ETESVS\svtas\loader\dataloader\opencv_dataloader.py
'''
import copy
import queue
from threading import Thread
from typing import Iterable, Optional, Sequence, Union, Dict, Any, List
import numpy as np
from .base_dataloader import BaseDataloader
from svtas.utils import AbstractBuildFactory, is_opencv_available

if is_opencv_available():
    import cv2

@AbstractBuildFactory.register('dataloader')
class OpencvDataloader(BaseDataloader):
    END_FLAG = False

    def __init__(self,
                 clip_seg_num: int,
                 sample_rate: int,
                 transform: Dict = None,
                 input_name: int | str = 0) -> None:
        super().__init__()
        self.input_name = input_name
        self.capture = cv2.VideoCapture(self.input_name)
        self.transform = AbstractBuildFactory.create_factory('dataset_transform').create(transform)
        self.clip_seg_num = clip_seg_num
        self.sample_rate = sample_rate
    
    @staticmethod
    def camera_buffer(capture, data_queue: queue.Queue):
        while True:
            ret, frame = capture.read()
            if not ret:
                capture.release()
                break
            data_queue.put(frame)
            cv2.imshow("Camera", frame)
            keyValue = cv2.waitKey(1)
            if keyValue & 0xFF == ord('q'):
                OpencvDataloader.END_FLAG = True
                break

    def _sample(self) -> Dict[str, Any]:
        data_queue = queue.Queue()
        camera_buffer_thread = Thread(target=self.camera_buffer, args=(self.capture, data_queue))
        camera_buffer_thread.start()
        while True:
            imgs = []
            for i in range(self.clip_seg_num * self.sample_rate):
                while data_queue.empty() and not OpencvDataloader.END_FLAG:
                    pass
                if OpencvDataloader.END_FLAG:
                    break 
                imgs.append(copy.deepcopy(data_queue.get()))
            if OpencvDataloader.END_FLAG:
                break
            data = imgs[::self.sample_rate]
            results = dict(
                imgs = data,
                raw_imgs = imgs,
                masks = np.ones((1, len(imgs))).astype(np.float32)
            )
            results = self.transform(results)
            results['imgs'] = results['imgs'].unsqueeze(0).numpy()
            yield results
        camera_buffer_thread.join()
        self.capture.release()

    def __iter__(self):
        return self._sample()

    def shuffle_dataloader(self, epoch) -> None:
        return super().shuffle_dataloader(epoch)