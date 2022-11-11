'''
Author       : Thyssen Wen
Date         : 2022-11-11 17:52:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-11 20:15:11
Description  : file content
FilePath     : /SVTAS/svtas/utils/stream_writer.py
'''
import tempfile
import os
import cv2
import ffmpy

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