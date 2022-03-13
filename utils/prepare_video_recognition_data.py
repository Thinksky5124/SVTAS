import json
import argparse
import os
import cv2
import numpy as np
import random
import math

from tqdm import tqdm

ignore_action_list = ["background", "None"]


def get_video_clip_list(label, background_id, neg_num, total_frames):
    clip_dict = {}
    video_ation_clip = []  # start_frame end_frame action_id
    video_background_clip = []  # start_frame end_frame action_id
    background_idx_list = []
    frame_index = 0
    background_list_idx = 0
    for action in label:
        start_frame = action["start_frame"]
        end_frame = action["end_frame"]
        if frame_index < start_frame:
            clip_start_frame = frame_index
            clip_end_frame = start_frame
            video_background_clip.append(
                [clip_start_frame, clip_end_frame, background_id])
            background_idx_list.append(background_list_idx)
            background_list_idx = background_list_idx + 1
        frame_index = start_frame
        clip_start_frame = start_frame
        clip_end_frame = end_frame
        video_ation_clip.append(
            [clip_start_frame, clip_end_frame, action["label_ids"]])
        frame_index = end_frame + 1
    if frame_index < total_frames:
        clip_start_frame = frame_index
        clip_end_frame = total_frames
        video_background_clip.append(
            [clip_start_frame, clip_end_frame, background_id])
        background_idx_list.append(background_list_idx)
    clip_dict["action_list"] = video_ation_clip
    clip_dict["background_list"] = video_background_clip
    clip_dict["len"] = len(background_idx_list)
    clip_dict["background_idx"] = background_idx_list
    return clip_dict


def video_split_to_clip(video_path, output_path_fix, video_name, label_fps,
                        sample_rate, video_clip_list, action_dict,
                        background_id, val_prob):
    result_dict = {}
    result_dict["rec_label_list"] = []
    result_dict["rec_val_label_list"] = []

    output_path_fix = os.path.join(output_path_fix, video_name)
    isExists = os.path.exists(output_path_fix)
    if not isExists:
        os.makedirs(output_path_fix)

    # read video
    video_capture = cv2.VideoCapture(video_path)

    # FPS
    fps = video_capture.get(5)
    if label_fps != fps:
        raise ImportError("label fps not match video fps!")

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
        start_frame = clip_info[0]
        end_frame = clip_info[1]
        action_id = clip_info[2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        if action_id == background_id:
            label_action_name = "background"
        else:
            label_action_name = action_dict[action_id]
        # prepare path fix
        output_path = os.path.join(
            output_path_fix,
            str(start_frame) + "_" + str(end_frame) + "_" + label_action_name +
            ".mp4")
        v = cv2.VideoWriter(output_path, fourcc, fps, size)
        for frame_index in range(start_frame, end_frame + 1):
            if frame_index < videolen:
                bgr_image = Frames[frame_index]
                v.write(bgr_image)
                # caculate BGR std and mean
                if random.random() >= sample_rate:
                    norm_imgs = np.reshape(bgr_image, (-1, 3)) / 255.0
                    action_mean = list(np.mean(norm_imgs, axis=0))
                    action_std = list(np.std(norm_imgs, axis=0))
                    video_mean.append(action_mean)
                    video_std.append(action_std)
        str_info = output_path + " " + str(action_id)
        result_dict["rec_label_list"].append(str_info)
        if random.random() < val_prob:
            result_dict["rec_val_label_list"].append(str_info)
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

    fps = label["fps"]

    # nomalization param
    mean = []
    std = []

    output_rec_txt_path = os.path.join(args.out_path, "recognition_label.txt")
    output_rec_val_txt_path = os.path.join(args.out_path, "recognition_val.txt")

    output_path_fix = os.path.join(args.out_path, "split_videos")
    isExists = os.path.exists(output_path_fix)
    if not isExists:
        os.makedirs(output_path_fix)
    rec_label_list = []
    rec_val_label_list = []
    clips_dict = {}
    for video_label in tqdm(label["gts"], desc='label collect:'):
        video_name = video_label["url"].split(".")[0]
        temp_dict = get_video_clip_list(video_label["actions"], background_id,
                                        args.negative_sample_num,
                                        video_label["total_frames"])
        clips_dict[video_name] = temp_dict

    clips_dict = resample_background(clips_dict, background_id,
                                     args.negative_sample_num)

    for video_label in tqdm(label["gts"], desc='video split:'):
        video_name = video_label["url"].split(".")[0]
        video_path = os.path.join(args.video_path, video_label["url"])
        video_clip_list = clips_dict[video_name]["list"]
        result_dict = video_split_to_clip(video_path, output_path_fix,
                                          video_name, fps, args.sample_rate,
                                          video_clip_list, id2class_map,
                                          background_id, args.validation_prob)
        label_list = result_dict["rec_label_list"]
        val_list = result_dict["rec_val_label_list"]
        mean = mean + result_dict["mean"]
        std = std + result_dict["std"]
        rec_label_list = rec_label_list + label_list
        rec_val_label_list = rec_val_label_list + val_list

    recog_content = [line + "\n" for line in rec_label_list]
    f = open(output_rec_txt_path, "w")
    f.writelines(recog_content)
    f.close()

    recog_content = [line + "\n" for line in rec_val_label_list]
    f = open(output_rec_val_txt_path, "w")
    f.writelines(recog_content)
    f.close()

    total_mean = list(np.mean(np.array(mean), axis=0))[::-1]
    total_std = list(np.mean(np.array(std), axis=0))[::-1]
    print("mean RGB :", total_mean, "std RGB :", total_std)


if __name__ == "__main__":
    main()
