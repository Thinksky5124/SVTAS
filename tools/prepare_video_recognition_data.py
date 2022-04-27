'''
Author: Thyssen Wen
Date: 2022-03-16 20:52:46
LastEditors: Thyssen Wen
LastEditTime: 2022-04-26 21:34:35
Description: prepare video recognition data and compute image std and mean script
FilePath: /ETESVS/utils/prepare_video_recognition_data.py
'''
import json
import argparse
import os
import cv2
import decord as de
import numpy as np
import random
import math

from tqdm import tqdm

ignore_action_list = ["background", "None", "STL"]


def get_video_clip_list(label, background_id, fps, total_frames):
    clip_dict = {}
    video_ation_clip = []  # start_frame end_frame action_id
    video_background_clip = []  # start_frame end_frame action_id
    background_idx_list = []
    frame_index = 0
    background_list_idx = 0
    for action in label:
        start_frame = int(np.floor(action["segment"][0] * fps))
        end_frame = int(np.floor(action["segment"][1] * fps)) + 1
        if frame_index < start_frame:
            clip_start_frame = frame_index
            clip_end_frame = start_frame
            video_background_clip.append(
                [clip_start_frame, clip_end_frame, "None"])
            background_idx_list.append(background_list_idx)
            background_list_idx = background_list_idx + 1
        frame_index = start_frame
        clip_start_frame = start_frame
        clip_end_frame = end_frame
        video_ation_clip.append(
            [clip_start_frame, clip_end_frame, action["label"]])
        frame_index = end_frame + 1
    if frame_index < total_frames:
        clip_start_frame = frame_index
        clip_end_frame = total_frames
        video_background_clip.append(
            [clip_start_frame, clip_end_frame, "None"])
        background_idx_list.append(background_list_idx)
    clip_dict["action_list"] = video_ation_clip
    clip_dict["background_list"] = video_background_clip
    clip_dict["len"] = len(background_idx_list)
    clip_dict["background_idx"] = background_idx_list
    return clip_dict

def caculate_video_std_mean(video_path, sample_rate, label_fps, dataset_type):
    result_dict = {}
    # read video
    if dataset_type in ['gtea', '50salads']:
        video_capture = de.VideoReader(video_path)
    elif dataset_type in ['thumos14', 'egtea']:
        video_capture = de.VideoReader(video_path + ".mp4")
    elif dataset_type in ['breakfast']:
        video_ptr = video_path.split('/')[-1].split('.')[0].split('_')
        video_prefix = '/'.join(video_path.split('/')[:-1])
        file_name = ''
        for j in range(2):
            if video_ptr[j] == 'stereo01':
                video_ptr[j] = 'stereo'
            file_name = file_name + video_ptr[j] + '/'
        file_name = file_name + video_ptr[2] + '_' + video_ptr[3]
        if 'stereo' in file_name:
            file_name = file_name + '_ch0'
        video_path = os.path.join(video_prefix, file_name)
        video_capture = de.VideoReader(video_path + ".avi")
    else:
        raise NotImplementedError
    
    fps = label_fps

    videolen = len(video_capture)

    # store mean std
    video_mean = []
    video_std = []
    frames_select = random.sample(range(videolen - 1), 10)
    frames_select.sort()
    if len(frames_select) > 0:
        bgr_image = video_capture.get_batch(frames_select).asnumpy()
        # caculate BGR std and mean
        norm_imgs = np.reshape(bgr_image, (-1, 3)) / 255.0
        action_mean = list(np.mean(norm_imgs, axis=0))
        action_std = list(np.std(norm_imgs, axis=0))
        video_mean.append(action_mean)
        video_std.append(action_std)
    else:
        bgr_image = video_capture.get_batch([(videolen - 1)//2]).asnumpy()
        # caculate BGR std and mean
        norm_imgs = np.reshape(bgr_image, (-1, 3)) / 255.0
        action_mean = list(np.mean(norm_imgs, axis=0))
        action_std = list(np.std(norm_imgs, axis=0))
        video_mean.append(action_mean)
        video_std.append(action_std)

    result_dict["mean"] = video_mean
    result_dict["std"] = video_std
    return result_dict

def video_split_to_clip(video_path, output_path_fix, video_name, label_fps,
                        sample_rate, video_clip_list, action_dict,
                        background_id, val_prob, only_norm_flag,
                        dataset_type):
    result_dict = {}
    result_dict["rec_label_list"] = []
    result_dict["rec_val_label_list"] = []

    output_path_fix = os.path.join(output_path_fix, video_name)
    
    if only_norm_flag is False:
        isExists = os.path.exists(output_path_fix)
        if not isExists:
            os.makedirs(output_path_fix)

    # read video
    if dataset_type in ['gtea', '50salads', 'thumos14', 'egtea']:
        video_capture = cv2.VideoCapture(video_path)
    elif dataset_type in ['breakfast']:
        video_ptr = video_path.split('/')[-1].split('.')[0].split('_')
        video_prefix = '/'.join(video_path.split('/')[:-1])
        file_name = ''
        for j in range(2):
            if video_ptr[j] == 'stereo01':
                video_ptr[j] = 'stereo'
            file_name = file_name + video_ptr[j] + '/'
        file_name = file_name + video_ptr[2] + '_' + video_ptr[3]
        if 'stereo' in file_name:
            file_name = file_name + '_ch0'
        video_path = os.path.join(video_prefix, file_name)
        video_capture = cv2.VideoCapture(video_path + ".avi")
    else:
        raise NotImplementedError

    # FPS
    # fps = video_capture.get(5)
    # if label_fps != fps:
    #     raise ImportError("label fps not match video fps!")
    fps = label_fps

    # get video height width
    size = (int(video_capture.get(3)), int(video_capture.get(4)))

    videolen = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    Frames = []
    for i in range(videolen):
        ret, frame = video_capture.read()
        # maybe first frame is empty
        if ret == False:
            continue
        img = frame
        Frames.append(img)

    # store mean std
    video_mean = []
    video_std = []
    for clip_info in tqdm(video_clip_list, desc=video_name):
        start_frame = int(np.floor(clip_info[0]))
        end_frame = int(np.floor(clip_info[1])) + 1
        action_name = clip_info[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if action_name in ignore_action_list:
            if dataset_type == "gtea":
                action_name = "background"
            elif dataset_type == "50salads":
                action_name = "action_end"
            elif dataset_type == "breakfast":
                action_name = "SIL"
            else:
                action_name = "None"
        else:
            label_action_name = action_name
        # prepare path fix
        output_path = os.path.join(
            output_path_fix,
            str(start_frame) + "_" + str(end_frame) + "_" + label_action_name +
            ".mp4")
        if only_norm_flag is False:
            v = cv2.VideoWriter(output_path, fourcc, fps, size)
        for frame_index in range(start_frame, end_frame + 1):
            if frame_index < videolen:
                bgr_image = Frames[frame_index]
                if only_norm_flag is False:
                    v.write(bgr_image)
                # caculate BGR std and mean
                if random.random() >= sample_rate:
                    norm_imgs = np.reshape(bgr_image, (-1, 3)) / 255.0
                    action_mean = list(np.mean(norm_imgs, axis=0))
                    action_std = list(np.std(norm_imgs, axis=0))
                    video_mean.append(action_mean)
                    video_std.append(action_std)

        str_info = output_path + " " + str(action_dict[action_name])
        result_dict["rec_label_list"].append(str_info)
        if random.random() < val_prob:
            result_dict["rec_val_label_list"].append(str_info)
            
        if only_norm_flag is False:
            v.release()
    video_capture.release()

    result_dict["mean"] = video_mean
    result_dict["std"] = video_std
    return result_dict


def resample_background(video_clip_dict, background_id, neg_num):
    clips_dict = {}
    key_list = list(video_clip_dict.keys())
    avg_sample = math.floor(neg_num / len(key_list))
    cnt_neg = 0
    if neg_num > 0:
        if avg_sample >= 1:
            for key in key_list:
                temp_dict = {}
                sample_idx = video_clip_dict[key]["background_idx"]
                random.shuffle(sample_idx)
                sample_idx = sample_idx[:int(avg_sample)]
                video_clip_dict["sampled"] = set(sample_idx)
                video_background_clip = []
                for idx in sample_idx:
                    video_background_clip.append(
                        video_clip_dict[key]["background_list"][idx])
                video_ation_clip = video_clip_dict[key]["action_list"]
                temp_dict["list"] = video_ation_clip + video_background_clip
                cnt_neg = cnt_neg + avg_sample
                clips_dict[key] = temp_dict

        if cnt_neg < neg_num:
            random.shuffle(key_list)
            for key in key_list:
                if key not in clips_dict.keys():
                    temp_dict = {}
                    cnt_neg = cnt_neg + video_clip_dict[key]["len"]
                    if cnt_neg > neg_num:
                        temp_dict["list"] = video_clip_dict[key]["action_list"]
                    else:
                        video_background_clip = video_clip_dict[key][
                            "background_list"]
                        video_ation_clip = video_clip_dict[key]["action_list"]
                        temp_dict[
                            "list"] = video_ation_clip + video_background_clip
                    clips_dict[key] = temp_dict
                else:
                    add_cnt_neg = cnt_neg + video_clip_dict[key]["len"]
                    if cnt_neg < neg_num and add_cnt_neg < neg_num:
                        sample_idx = video_clip_dict[key]["background_idx"]
                        random.shuffle(sample_idx)
                        unsample_idx = set(sample_idx)
                        sample_idx = (unsample_idx - video_clip_dict["sampled"])
                        video_background_clip = []
                        for idx in sample_idx:
                            video_background_clip.append(
                                video_clip_dict[key]["background_list"][idx])
                        temp_dict[
                            "list"] = temp_dict["list"] + video_background_clip
                        cnt_neg = cnt_neg + len(sample_idx)
                    elif cnt_neg < neg_num and add_cnt_neg > neg_num:
                        sample_idx = video_clip_dict[key]["background_idx"]
                        random.shuffle(sample_idx)
                        unsample_idx = set(sample_idx)
                        sample_idx = list(unsample_idx -
                                          video_clip_dict["sampled"])[:(
                                              neg_num - cnt_neg)]
                        video_background_clip = []
                        for idx in sample_idx:
                            video_background_clip.append(
                                video_clip_dict[key]["background_list"][idx])
                        temp_dict[
                            "list"] = temp_dict["list"] + video_background_clip
                        cnt_neg = cnt_neg + len(sample_idx)
                    elif cnt_neg > neg_num:
                        break
    else:
        for key in key_list:
            temp_dict = {}
            video_ation_clip = video_clip_dict[key]["action_list"]
            temp_dict["list"] = video_ation_clip
            clips_dict[key] = temp_dict
    return clips_dict


def load_action_dict(data_path):
    mapping_txt_path = os.path.join(data_path, "mapping.txt")
    with open(mapping_txt_path, "r", encoding='utf-8') as f:
        actions = f.read().split("\n")[:-1]

    id2class_map = dict()
    class2id_map = dict()
    for a in actions:
        id2class_map[int(a.split()[0])] = a.split()[1]
        class2id_map[a.split()[1]] = int(a.split()[0])

    return id2class_map, class2id_map


def get_arguments():
    """
    parse all the arguments from command line inteface
    return a list of parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="convert segmentation and localization label")
    parser.add_argument("label_path", type=str, help="path of a label file")
    parser.add_argument(
        "video_path",
        type=str,
        help="path of video.",
    )
    parser.add_argument(
        "out_path",
        type=str,
        help="path of output file.",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=0.5,
        help="sample rate of computing mean and std.",
    )
    parser.add_argument(
        "--validation_prob",
        type=float,
        default=0.3,
        help="probabilty of validation set.",
    )
    parser.add_argument(
        "--negative_sample_num",
        type=int,
        default=100,
        help="sample rate of computing mean and std.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="Convert label fps.",
    )
    parser.add_argument(
        "--only_norm",
        type=bool,
        default=True,
        help="Only caculate video RGB mean and std.",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="gtea",
        help="sample rate of computing mean and std.",
    )
    return parser.parse_args()


def main():
    args = get_arguments()

    with open(args.label_path, 'r', encoding='utf-8') as json_file:
        label = json.load(json_file)

    id2class_map, class2id_map = load_action_dict(args.out_path)

    background_id = None
    for ignore_action in ignore_action_list:
        if ignore_action in class2id_map.keys():
            background_id = class2id_map[ignore_action]
    if background_id is None:
        background_id = len(id2class_map.keys())

    fps = args.fps
    only_norm_flag = args.only_norm
    # nomalization param
    mean = []
    std = []

    output_rec_txt_path = os.path.join(args.out_path, "val_list.txt")
    output_rec_val_txt_path = os.path.join(args.out_path, "test_list.txt")

    output_path_fix = os.path.join(args.out_path, "split_videos")
    
    if only_norm_flag is False:
        isExists = os.path.exists(output_path_fix)
        if not isExists:
            os.makedirs(output_path_fix)
    rec_label_list = []
    rec_val_label_list = []
    clips_dict = {}
    for vid, video_label in tqdm(label["database"].items(), desc='label collect:'):
        video_name = vid.split(".")[0]
        temp_dict = get_video_clip_list(video_label["annotations"], background_id,
                                        args.fps,
                                        video_label["frames"])
        clips_dict[video_name] = temp_dict

    clips_dict = resample_background(clips_dict, background_id,
                                     args.negative_sample_num)

    for vid, video_label in tqdm(label["database"].items(), desc='video split:'):
        video_name = vid.split(".")[0]
        video_path = os.path.join(args.video_path, vid)
        video_clip_list = clips_dict[video_name]["list"]
        if only_norm_flag is False:
            result_dict = video_split_to_clip(video_path, output_path_fix,
                                            video_name, fps, args.sample_rate,
                                            video_clip_list, class2id_map,
                                            background_id, args.validation_prob,
                                            only_norm_flag, args.dataset_type)
            label_list = result_dict["rec_label_list"]
            val_list = result_dict["rec_val_label_list"]
            mean = mean + result_dict["mean"]
            std = std + result_dict["std"]
            rec_label_list = rec_label_list + label_list
            rec_val_label_list = rec_val_label_list + val_list
        elif only_norm_flag is True:
            result_dict = caculate_video_std_mean(video_path, args.sample_rate, fps, args.dataset_type)
            mean = mean + result_dict["mean"]
            std = std + result_dict["std"]
        else:
            break

    if only_norm_flag is False:
        recog_content = [line + "\n" for line in rec_label_list]
        f = open(output_rec_txt_path, "w")
        f.writelines(recog_content)
        f.close()

        recog_content = [line + "\n" for line in rec_val_label_list]
        f = open(output_rec_val_txt_path, "w")
        f.writelines(recog_content)
        f.close()

    if only_norm_flag is False:
        total_mean = list(np.mean(np.array(mean), axis=0))[::-1]
        total_std = list(np.mean(np.array(std), axis=0))[::-1]
    else:
        total_mean = list(np.mean(np.array(mean), axis=0))
        total_std = list(np.mean(np.array(std), axis=0))
    print("mean RGB :", total_mean, "std RGB :", total_std)


if __name__ == "__main__":
    main()
