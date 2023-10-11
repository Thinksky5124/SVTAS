'''
Author       : Thyssen Wen
Date         : 2022-12-23 21:44:30
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-11 20:55:33
Description  : file content
FilePath     : /SVTAS/svtas/utils/cam/match_fn.py
'''
import cv2
import numpy as np
from ..package_utils import is_pytorch_grad_cam_available

if is_pytorch_grad_cam_available():
    from pytorch_grad_cam.utils.image import show_cam_on_image, \
        preprocess_image

def rgb_stream_match_fn(data_dict, grayscale_cam):
    cam_image_list = []
    for batch_id in range(len(data_dict['raw_imgs'])):
        batch_image_list = []
        for sample_id in range(len(data_dict['raw_imgs'][batch_id])):
            rgb_img = cv2.cvtColor(np.asarray(data_dict['raw_imgs'][batch_id][sample_id]), cv2.COLOR_RGB2BGR)[:, :, ::-1]
            rgb_img = np.float32(rgb_img) / 255
            rgb_img = cv2.resize(rgb_img, (grayscale_cam.shape[-1], grayscale_cam.shape[-2]))
            grayscale_cam_sample = grayscale_cam[batch_id * len(data_dict['raw_imgs'][batch_id]) + sample_id, :]

            cam_image = show_cam_on_image(rgb_img, grayscale_cam_sample)
            batch_image_list.append(np.expand_dims(cam_image, 0))
        batch_image = np.expand_dims(np.concatenate(batch_image_list, 0), 0)
        cam_image_list.append(batch_image)
    cam_images = np.concatenate(cam_image_list, 0)
    return cam_images

def feature_batch_match_fn(data_dict, grayscale_cam):
    cam_image_list = []
    for batch_id in range(len(data_dict['raw_feature'])):
        gray_img = np.expand_dims(data_dict['raw_feature'][batch_id], -1)
        norm_img = np.zeros(gray_img.shape)
        # get black image
        cv2.normalize(norm_img , norm_img, 0, 255, cv2.NORM_MINMAX)
        norm_img = np.asarray(norm_img, dtype=np.uint8)
        heat_img = cv2.applyColorMap(norm_img, cv2.COLORMAP_JET)
        heat_img = cv2.cvtColor(heat_img, cv2.COLOR_BGR2RGB)
        heat_img = np.float32(heat_img) / 255
        heat_img = cv2.resize(heat_img, (grayscale_cam.shape[-2], grayscale_cam.shape[-1]))
        grayscale_cam_sample = grayscale_cam[batch_id, :].transpose(-1, -2)

        cam_image = show_cam_on_image(heat_img, grayscale_cam_sample)
        batch_image = np.expand_dims(np.expand_dims(cam_image, 0), 0)
        cam_image_list.append(batch_image)
    cam_images = np.concatenate(cam_image_list, 0)
    return cam_images