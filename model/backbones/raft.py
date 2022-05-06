'''
Author       : Thyssen Wen
Date         : 2022-05-06 13:44:50
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-06 14:41:57
Description  : RAFT ref:https://github.com/princeton-vl/RAFT
FilePath     : /ETESVS/model/backbones/raft.py
'''
'''
    Reference: https://github.com/princeton-vl/RAFT/tree/25eb2ac723c36865c636c9d1f497af8023981868
    Modified by Vladimir Iashin for github.com/v-iashin/video_features
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.raft.update import BasicUpdateBlock, SmallUpdateBlock
from .utils.raft.extractor import BasicEncoder, SmallEncoder
from .utils.raft.corr import CorrBlock, AlternateCorrBlock
from .utils.raft.utils import bilinear_sampler, coords_grid, upflow8

from mmcv.runner import load_checkpoint
from utils.logger import get_logger
from ..builder import BACKBONES

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass

class InputPadder:
    """ Pads images such that dimensions are divisible by 8"""
    def __init__(self, dims, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // 8) + 1) * 8 - self.ht) % 8
        pad_wd = (((self.wd // 8) + 1) * 8 - self.wd) % 8
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, input):
        b, t, c, h, w = input.shape
        input = torch.reshape(input, shape=[-1] + list(input.shape[-3:]))
        output = F.pad(input, self._pad, mode='replicate')
        output = torch.reshape(output, shape=[-1, t] + list(output.shape[-3:]))
        return output

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

@BACKBONES.register()
class RAFT(nn.Module):

    # (v-iashin) def __init__(self, model_is_small):
    def __init__(self,
                 pretrained=None,
                 extract_mode=True,
                 freeze=True,
                 model_is_small=False,
                 alternate_corr=False,
                 mixed_precision=False,
                 dropout=0,
                 mode='sintel'):
        super(RAFT, self).__init__()
        self.dropout = dropout
        self.alternate_corr = alternate_corr
        self.model_is_small = model_is_small
        self.mixed_precision = mixed_precision
        self.pretrained = pretrained
        self.extract_mode = extract_mode
        self.freeze = freeze
        self.mode = mode

        self.memory_frame = None

        if self.model_is_small:
            self.corr_levels = 4
            self.corr_radius = 3
            self.hidden_dim = 96
            self.context_dim = 64
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        else:
            self.corr_levels = 4
            self.corr_radius = 4
            self.hidden_dim = 128
            self.context_dim = 128
            self.cnet_out_dim = self.hidden_dim + self.context_dim

        # feature network, context network, and update block
        if self.model_is_small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=self.dropout)
            self.cnet = SmallEncoder(output_dim=self.cnet_out_dim, norm_fn='none', dropout=self.dropout)
            self.update_block = SmallUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=self.dropout)
            self.cnet = BasicEncoder(output_dim=self.cnet_out_dim, norm_fn='batch', dropout=self.dropout)
            self.update_block = BasicUpdateBlock(
                self.corr_levels, self.corr_radius, hidden_dim=self.hidden_dim
            )

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)

    def init_weights(self, child_model=False, revise_keys=[(r'^module\.', '')]):
        if child_model is False:
            if isinstance(self.pretrained, str):
                logger = get_logger("ETESVS")
                load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=revise_keys)
                if self.freeze is True:
                    self.eval()
                    for param in self.parameters():
                        param.requires_grad = False
            elif self.pretrained is None:
                for m in self.modules():
                    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                        nn.init.kaiming_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)
            else:
                raise TypeError('pretrained must be a str or None')
        else:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
    
    def _clear_memory_buffer(self):
        self.memory_frame = None

    def pre_precessing(self, x):
        self.padder = InputPadder(x.shape)
        x = self.padder.pad(x)
        # x.shape [N, T, C, H, W]
        if self.memory_frame is None:
            # img1.shape [N, (T - 1), C, H, W]
            img1 = x[:, :-1]
            img1 = torch.reshape(img1, shape=[-1] + list(img1.shape[-3:]))
            # img2.shape [N, (T - 1), C, H, W]
            img2 = x[:, 1:]
            img2 = torch.reshape(img2, shape=[-1] + list(img2.shape[-3:]))
            self.memory_frame = x[:, -2:-1].detach().clone()
        else:
            # img1.shape [N, T, C, H, W]
            img1 = torch.cat([self.memory_frame, x[:, :-1]], dim=1)
            img1 = torch.reshape(img1, shape=[-1] + list(img1.shape[-3:]))
            # img2.shape [N, T, C, H, W]
            img2 = x
            img2 = torch.reshape(img2, shape=[-1] + list(img2.shape[-3:]))
            self.memory_frame = x[:, -2:-1].detach().clone()
        
        temporal_len = img1.shape[0]
        height, width = x.shape[-2:]
        orig_size = (int(height), int(width))

        input_size = orig_size
        
        return img1, img2, input_size, orig_size, temporal_len
    
    def post_precessing(self, flow, input_size, orig_size, temporal_len):
        flow = self.padder.unpad(flow)
        # [N, T, C, H, W]
        refine_flow = torch.reshape(flow, shape=[-1] + [temporal_len, 2] + list(flow.shape[-2:]))

        return refine_flow

    # (v-iashin) def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
    def forward(self, image1, image2, iters=20, flow_init=None, upsample=True):
        """ Estimate optical flow between pair of frames """

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with autocast(enabled=self.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        if self.alternate_corr:
            corr_fn = AlternateCorrBlock(fmap1, fmap2, radius=self.corr_radius)
        else:
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.corr_radius)

        # run the context network
        with autocast(enabled=self.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)

            flow_predictions.append(flow_up)

        if self.extract_mode:
            # (v-iashin) return coords1 - coords0, flow_up
            return flow_up

        return flow_predictions