'''
Author       : Thyssen Wen
Date         : 2022-11-11 09:48:15
LastEditors  : Thyssen Wen
LastEditTime : 2022-11-12 15:19:49
Description  : file content
FilePath     : /SVTAS/svtas/model/post_precessings/mvs_res_post_processing.py
'''
import numpy as np
import torch
from ...utils.flow_vis import make_colorwheel
from ..builder import POSTPRECESSING
from ...loader.transform.transform import VideoStreamTransform
from ...utils.stream_writer import VideoStreamWriter

@POSTPRECESSING.register()
class MVsResPostProcessing():
    def __init__(self,
                 sliding_window,
                 mvs_post_transforms=[dict(Clamp = dict(min_val=-20, max_val=20)),
                                      dict(ToUInt8 = None)],
                 res_post_transforms=[dict(Clamp = dict(min_val=0, max_val=255)),
                                      dict(DtypeToUInt8 = None)],
                 fps=15,
                 need_visualize=False,
                 ignore_index=-100):
        self.sliding_window = sliding_window
        self.fps = fps
        self.need_visualize = need_visualize
        self.mvs_post_transforms = VideoStreamTransform(mvs_post_transforms)
        self.res_post_transforms = VideoStreamTransform(res_post_transforms)
        self.init_flag = False
        self.colorwheel = make_colorwheel()  # shape [55x3]
        self.ignore_index = ignore_index
    
    def init_scores(self, sliding_num, batch_size):
        self.mvs_img_list = []
        self.res_img_list = []
        self.mvs_visual_list = []
        self.video_gt = []
        self.init_flag = True
    
    def _update_mvs_imgs(self, mvs_imgs):
        # flow_imgs [N T C H W]
        for bs in range(mvs_imgs.shape[0]):
            results = {}
            results['imgs'] = mvs_imgs[bs, :]
            flows = self.mvs_post_transforms(results)['imgs']
            flows = flows.cpu().permute(0, 2, 3, 1).numpy()
            flows = np.concatenate([flows, np.zeros_like(flows[:, :, :, 0:1])], axis=-1)
            if len(self.mvs_img_list) < (bs + 1):
                self.mvs_img_list.append(VideoStreamWriter(self.fps))
            self.mvs_img_list[bs].stream_write(flows)
            
            if self.need_visualize:
                u = mvs_imgs[bs, :].cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 0]
                v = mvs_imgs[bs, :].cpu().permute(0, 2, 3, 1).numpy()[:, :, :, 1]
                rad = np.sqrt(np.square(u) + np.square(v))
                rad_max = np.max(rad)
                epsilon = 1e-5
                u = u / (rad_max + epsilon)
                v = v / (rad_max + epsilon)
                
                flows_image = np.zeros((u.shape[0], u.shape[1], u.shape[2], 3), np.uint8)

                ncols = self.colorwheel.shape[0]
                rad = np.sqrt(np.square(u) + np.square(v))
                a = np.arctan2(-v, -u)/np.pi
                fk = (a + 1) / 2 * (ncols - 1)
                k0 = np.floor(fk).astype(np.int32)
                k1 = k0 + 1
                k1[k1 == ncols] = 0
                f = fk - k0
                for i in range(self.colorwheel.shape[1]):
                    tmp = self.colorwheel[:, i]
                    col0 = tmp[k0] / 255.0
                    col1 = tmp[k1] / 255.0
                    col = (1 - f) * col0 + f * col1
                    idx = (rad <= 1)
                    col[idx]  = 1 - rad[idx] * (1-col[idx])
                    col[~idx] = col[~idx] * 0.75   # out of range
                    # Note the 2-i => BGR instead of RGB
                    ch_idx = 2 - i
                    flows_image[:, :, :, ch_idx] = np.floor(255 * col)
                if len(self.mvs_visual_list) < (bs + 1):
                    self.mvs_visual_list.append(VideoStreamWriter(self.fps))
                self.mvs_img_list[bs].stream_write(flows_image)

    def _update_res_imgs(self, res_imgs):
        for bs in range(res_imgs.shape[0]):
            results = {}
            results['imgs'] = res_imgs[bs, :]
            res = self.res_post_transforms(results)['imgs']
            res = res.cpu().permute(0, 2, 3, 1).numpy()
            if len(self.res_img_list) < (bs + 1):
                self.res_img_list.append(VideoStreamWriter(self.fps))
            self.res_img_list[bs].stream_write(res)

    def update(self, imgs_dict, gt, idx):
        # seg_scores [stage_num N C T]
        # gt [N T]
        self.video_gt.append(gt[:, 0:self.sliding_window].detach().cpu().numpy().copy())
        for k ,v in imgs_dict.items():
            if k == "flows":
                self._update_mvs_imgs(v)
            if k == "res":
                self._update_res_imgs(v)


    def output(self):
        # save flow imgs
        flow_imgs_list = []
        need_res = False
        if len(self.res_img_list) > 0:
            need_res = True
            res_imgs_list = []

        video_gt = np.concatenate(self.video_gt, axis=1)
        if self.need_visualize:
            flow_visual_imgs_list = []

        for bs in range(len(self.mvs_img_list)):
            index = np.where(video_gt[bs, :] == self.ignore_index)
            ignore_start = min(list(index[0]) + [video_gt.shape[-1]])
            self.mvs_img_list[bs].dump()
            flow_imgs_list.append({"writer":self.mvs_img_list[bs], "len":ignore_start})
            if self.need_visualize:
                self.mvs_visual_list[bs].dump()
                flow_visual_imgs_list.append({"writer":self.mvs_visual_list[bs], "len":ignore_start})
            if need_res:
                self.res_img_list[bs].dump()
                res_imgs_list.append({"writer":self.res_img_list[bs], "len":ignore_start})
                
        if self.need_visualize:
            if need_res:
                return flow_imgs_list, res_imgs_list, flow_visual_imgs_list
            else:
                return flow_imgs_list, flow_visual_imgs_list
        else:
            if need_res:
                return flow_imgs_list, res_imgs_list
            else:
                return [flow_imgs_list]
