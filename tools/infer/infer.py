'''
Author       : Thyssen Wen
Date         : 2022-10-21 16:30:17
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-22 10:53:11
Description  : Multi Device Infer Script with visualize program
FilePath     : /SVTAS/tools/infer/infer.py
'''
import os
import cv2
import copy
import queue
import numpy as np
import argparse
import onnxruntime
from PIL import Image
from cv2 import getTickCount, getTickFrequency

def load_capture(args):
    capture = cv2.VideoCapture(args.input)
    return capture

def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit


    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette

def draw_action_label(img, palette, action_dict, label):
    fix_buffer = 12
    for i in range(len(label)):
        k = label[i]
        color_plate = (int(palette[k][2]), int(palette[k][1]), int(palette[k][0]))
        img = cv2.rectangle(img, (5, 15 + fix_buffer * i), (25, 5 + fix_buffer * i), color_plate, thickness=-1)
        cv2.putText(img, action_dict[k], (30, 12 + fix_buffer * i), cv2.FONT_HERSHEY_COMPLEX, 0.25, color_plate, 1)
        
    return img


def label_arr2img(label_queue, palette):
    data = list(copy.deepcopy(label_queue.queue))
    array = np.array(data).transpose()
    arr = array.astype(np.uint8)
    arr = np.tile(arr, (20, 1))
    img = Image.fromarray(arr)
    img = img.convert("P")
    img.putpalette(palette)
    return img

def parse_args():
    parser = argparse.ArgumentParser("SVTAS infer script")
    parser.add_argument('-m',
                        '--model',
                        type=str,
                        default='model.onnx',
                        help='onnx model path')
    parser.add_argument('-i',
                        '--input',
                        type=str,
                        default='example.mp4',
                        help='infer video path')
    parser.add_argument('-l',
                        '--label',
                        type=str,
                        default='mapping.txt',
                        help='infer video mapping label txt')
    parser.add_argument('-o',
                        '--output',
                        type=str,
                        default='example_infer.mp4',
                        help='infer video output path')
    parser.add_argument('--visualize',
                        action="store_true",
                        help='wheather visualize video when infer')
    parser.add_argument('--sample_rate',
                        type=int,
                        default=4,
                        help='video sample rate')
    parser.add_argument('--clip_seg_num',
                        type=int,
                        default=8,
                        help='infer model windows size')
    parser.add_argument('--sliding_window',
                        type=int,
                        default=32,
                        help='infer model sliding windows size')   
    args = parser.parse_args()
    return args

def infer():
    args = parse_args()
    #! set mean and std
    memory_factor = 3
    mean = [[[0.551, 0.424, 0.179]]]
    std = [[[0.133, 0.141, 0.124]]]
    mean = np.array(mean)[:,:,::-1].transpose((2,0,1))
    std = np.array(std)[:,:,::-1].transpose((2,0,1))

    # load model
    ort_session = onnxruntime.InferenceSession(args.model)

    # load mapping label
    file_ptr = open(args.label, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    file_ptr.close()
    actions_dict = dict()
    for a in actions:
        actions_dict[int(a.split()[0])] = a.split()[1]
    palette = make_palette(len(actions_dict))

    # load video
    capture = load_capture(args)
    
    # get video width
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # get video height
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # get video fps
    fps = capture.get(cv2.CAP_PROP_FPS)

    # write video setting
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width, frame_height))

    # online infer looping
    data_queue = queue.Queue(maxsize=args.clip_seg_num * args.sample_rate)
    label_queue = queue.Queue(maxsize=args.clip_seg_num * args.sample_rate * memory_factor)
    infer_run_flag = True
    while data_queue.qsize() != (args.clip_seg_num * args.sample_rate):
        ret, frame = capture.read()
        if not ret:
            infer_run_flag = False
            break
        data_queue.put(frame)
        
    loop_start = getTickCount()
    while infer_run_flag:
        # preprocess
        data = list(copy.deepcopy(data_queue.queue))[::args.sample_rate]
        imgs = []
        for img in data:
            img = cv2.resize(img, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[:,:,::-1].transpose((2,0,1))
            img = img.astype(np.float64) / 255
            img-=mean
            img/=std
            imgs.append(img)
        imgs = np.array(imgs)
        imgs = np.expand_dims(imgs, axis=0)
        masks = np.ones((1, args.clip_seg_num * args.sample_rate))
        input_data = dict(input_data=imgs.astype(np.float32), masks=masks.astype(np.float32))
        # model infer
        outputs = ort_session.run(None, input_data)
        if type(outputs) is not np.ndarray:
            outputs = outputs[-1]

        # post process
        loop_time = getTickCount() - loop_start
        total_time = loop_time / (getTickFrequency())
        chunk_fps = 1 / total_time
        for i in range(args.sliding_window):
            out_frame = data_queue.get()
            # add infer info
            cv2.putText(out_frame, "Prediction: " + actions_dict[np.argmax(outputs[0, -1, :, i])], (0, frame_height - 60), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(out_frame, "FPS: " + "{:.2f}".format(chunk_fps), (frame_width - 150, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
            if label_queue.full():
                label_queue.get()
            label_queue.put([np.argmax(outputs[0, -1, :, i])])
            label_img = label_arr2img(label_queue, palette).convert('RGB')
            past_width = int((label_img.size[0] / (args.clip_seg_num * args.sample_rate * memory_factor)) * (frame_width - 30))
            label_img = cv2.cvtColor(np.asarray(label_img),cv2.COLOR_RGB2BGR)
            label_img = cv2.resize(label_img, (past_width, 20))
            out_frame[(frame_height - 30):(frame_height - 10), 10:(10 + past_width), :] = label_img
            out_frame = cv2.rectangle(out_frame, (10 + past_width, frame_height - 10), (20 + past_width, frame_height - 30), (255, 255, 255), thickness=-1)
            cv2.putText(out_frame, "Current Frame", (max(past_width - 120, 0), frame_height - 40), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)

            data = list(copy.deepcopy(label_queue.queue))
            array = np.array(data).transpose()
            label = list(set(array[0, :].tolist()))
            out_frame = draw_action_label(out_frame, palette, actions_dict, label)
            
            # visualize program
            if args.visualize:
                cv2.imshow("real-time temporal action segmentation", out_frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    infer_run_flag = False
                    break
            out.write(out_frame)
            
        # load frame
        while not data_queue.full():
            ret, frame = capture.read()
            if not ret:
                infer_run_flag = False
                break
            data_queue.put(frame)

        loop_start = getTickCount()
    # save process
    capture.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    infer()