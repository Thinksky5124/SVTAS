'''
Author       : Thyssen Wen
Date         : 2022-11-11 17:52:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-23 11:41:16
Description  : file content
FilePath     : /SVTAS/svtas/utils/stream_writer.py
'''
import tempfile
import os
import cv2
import ffmpy
import queue
from .misc import label_arr2img, draw_action_label
import numpy as np
import copy

class StreamWriter(object):
    def __init__(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.concat_file_list = []
        self.dump_tem_file = None
    
    @classmethod
    def stream_write(self, data):
        pass
    
    @classmethod
    def save(self, path, len):
        pass

    @classmethod
    def dump(self, path):
        pass

class NPYStreamWriter(StreamWriter):
    def __init__(self):
        super().__init__()
        self.cnt = 0
    
    def stream_write(self, data):
        npy_path = os.path.join(self.tempdir.name, str(self.cnt) + '.npy')
        self.cnt = self.cnt + 1
        np.save(npy_path, data)
        self.concat_file_list.append(npy_path)
    
    def save(self, path, len):
        temp_npy = np.load(self.dump_tem_file)
        temp_npy = temp_npy[:, :len]
        np.save(path, temp_npy)

        self.cnt = 0
        self.tempdir.cleanup()

    def dump(self):        
        self.dump_tem_file = os.path.join(self.tempdir.name, 'temp.npy')
        
        temp = []
        for file_npy in self.concat_file_list:
            temp_data = np.load(file_npy, allow_pickle=True)
            temp.append(temp_data)
        temp = np.concatenate(temp, axis=-1)
        np.save(self.dump_tem_file, temp)

class VideoStreamWriter(StreamWriter):
    def __init__(self,
                 fps):
        super().__init__()
        self.fps = fps
        self.cnt = 0
    
    def stream_write(self, imgs):
        video_path = os.path.join(self.tempdir.name, str(self.cnt) + '.mp4')
        self.cnt = self.cnt + 1
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWrite = cv2.VideoWriter(video_path, fourcc, self.fps, (imgs.shape[-2], imgs.shape[-3]))

        for img in imgs:
            videoWrite.write(img)
        self.concat_file_list.append("file " + video_path)
    
    def save(self, path, len):
        raw_video = cv2.VideoCapture(self.dump_tem_file)
        frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWrite = cv2.VideoWriter(path, fourcc, self.fps, (frame_width, frame_height))

        count = 0
        while True:
            ret, img = raw_video.read()
            if ret:
                videoWrite.write(img)
                count = count + 1
            else:
                break
            if count >= len:
                break
        videoWrite.release()

        self.cnt = 0
        self.tempdir.cleanup()

    def dump(self):
        concat_file = os.path.join(self.tempdir.name, 'concat_list.txt')
        with open(concat_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(self.concat_file_list))
        
        self.dump_tem_file = os.path.join(self.tempdir.name, 'temp.mp4')
        
        ff = ffmpy.FFmpeg(
            global_options=['-f', 'concat', '-safe', '0'],
            inputs={concat_file: None},
            outputs={self.dump_tem_file: ['-c', 'copy']}
        )

        ff.run()

class CAMVideoStreamWriter(VideoStreamWriter):
    def __init__(self, fps, frame_height, frame_width, label_log_len=32, need_label=True):
        super().__init__(fps)
        self.label_log_len = label_log_len
        self.frame_height = frame_height
        self.frame_width = frame_width
        self.need_label = need_label
        
    def save(self, path, len, labels, preds, action_dict, palette):
        raw_video = cv2.VideoCapture(self.dump_tem_file)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        videoWrite = cv2.VideoWriter(path, fourcc, self.fps, (self.frame_width, self.frame_height))
        
        pred_queue = queue.Queue(maxsize=self.label_log_len)
        label_queue = queue.Queue(maxsize=self.label_log_len)
        count = 0
        while True:
            ret, img = raw_video.read()
            if ret:
                img = cv2.resize(img, (self.frame_width, self.frame_height))
                if self.need_label:
                    # add pred and gt info
                    cv2.putText(img, "Prediction: " + action_dict[preds[count]], (0, self.frame_height - 100), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                    cv2.putText(img, "Groundtruth: " + action_dict[labels[count]], (0, self.frame_height - 80), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
                    if pred_queue.full():
                        pred_queue.get()
                        label_queue.get()
                    pred_queue.put([preds[count]])
                    label_queue.put([labels[count]])
                    pred_img = label_arr2img(pred_queue, palette).convert('RGB')
                    label_img = label_arr2img(label_queue, palette).convert('RGB')
                    past_width = int((label_img.size[0] / 32) * (self.frame_width - 40))
                    pred_img = cv2.cvtColor(np.asarray(pred_img),cv2.COLOR_RGB2BGR)
                    label_img = cv2.cvtColor(np.asarray(label_img),cv2.COLOR_RGB2BGR)
                    pred_img = cv2.resize(pred_img, (past_width, 20))
                    label_img = cv2.resize(label_img, (past_width, 20))
                    cv2.putText(img, "Pr: ", (0, self.frame_height - 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    img[(self.frame_height - 50):(self.frame_height - 30), 30:(30 + past_width), :] = pred_img
                    cv2.putText(img, "GT: ", (0, self.frame_height - 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
                    img[(self.frame_height - 30):(self.frame_height - 10), 30:(30 + past_width), :] = label_img
                    # Line 1 prediction Line 2 groundtruth
                    img = cv2.rectangle(img, (20 + past_width, self.frame_height - 10), (30 + past_width, self.frame_height - 50), (255, 255, 255), thickness=-1)
                    cv2.line(img, (30, self.frame_height - 30), (30 + past_width, self.frame_height - 30), (255,255,255), 1)
                    cv2.putText(img, "Current Frame", (max(past_width - 110, 0), self.frame_height - 55), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

                    data_pred = list(copy.deepcopy(pred_queue.queue))
                    data_label = list(copy.deepcopy(label_queue.queue))
                    array_pred = np.array(data_pred).transpose()
                    array_label = np.array(data_label).transpose()
                    label = list(set(array_pred[0, :].tolist()) | set(array_label[0, :].tolist()))
                    img = draw_action_label(img, palette, action_dict, label)
                
                videoWrite.write(img)
                count = count + 1
            else:
                break
            if count >= len:
                break
        videoWrite.release()

        self.cnt = 0
        self.tempdir.cleanup()