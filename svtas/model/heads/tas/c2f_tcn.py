'''
Author       : Thyssen Wen
Date         : 2023-02-24 15:04:22
LastEditors  : Thyssen Wen
LastEditTime : 2023-02-24 15:39:43
Description  : C2F-TCN ref:https://github.com/dipika-singhania/C2F-TCN/blob/main/model.py
FilePath     : /SVTAS/svtas/model/heads/tas/c2f_tcn.py
'''
import torch.nn.functional as F
import torch.nn as nn
import torch
from functools import partial

from ...builder import HEADS
nonlinearity = partial(F.relu, inplace=True)

class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool1d(2), double_conv(in_ch, out_ch))

    def forward(self, x):
        x = self.max_pool_conv(x)
        return x

class up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="linear", align_corners=True)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2
            )

        self.conv = double_conv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = torch.tensor([x2.size()[2] - x1.size()[2]])

        x1 = F.pad(x1, [diff // 2, diff - diff //2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class TPPblock(nn.Module):
    def __init__(self, in_channels):
        super(TPPblock, self).__init__()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=3)
        self.pool3 = nn.MaxPool1d(kernel_size=5, stride=5)
        self.pool4 = nn.MaxPool1d(kernel_size=6, stride=6)

        self.conv = nn.Conv1d(
            in_channels=in_channels, out_channels=1, kernel_size=1, padding=0
        )

    def forward(self, x):
        self.in_channels, t = x.size(1), x.size(2)
        self.layer1 = F.upsample(
            self.conv(self.pool1(x)), size=t, mode="linear", align_corners=True
        )
        self.layer2 = F.upsample(
            self.conv(self.pool2(x)), size=t, mode="linear", align_corners=True
        )
        self.layer3 = F.upsample(
            self.conv(self.pool3(x)), size=t, mode="linear", align_corners=True
        )
        self.layer4 = F.upsample(
            self.conv(self.pool4(x)), size=t, mode="linear", align_corners=True
        )

        out = torch.cat([self.layer1, self.layer2,
                         self.layer3, self.layer4, x], 1)

        return out

@HEADS.register()
class C2F_TCN(nn.Module):
    '''
        Features are extracted at the last layer of decoder. 
    '''
    def __init__(self,
                 n_channels,
                 num_classes,
                 ensem_weights = [1, 1, 1, 1, 0, 0],
                 sample_rate=1):
        super(C2F_TCN, self).__init__()
        self.sample_rate = sample_rate
        self.ensem_weights = ensem_weights

        self.inc = inconv(n_channels, 256)
        self.down1 = down(256, 256)
        self.down2 = down(256, 256)
        self.down3 = down(256, 128)
        self.down4 = down(128, 128)
        self.down5 = down(128, 128)
        self.down6 = down(128, 128)
        self.up = up(260, 128)
        self.outcc0 = outconv(128, num_classes)
        self.up0 = up(256, 128)
        self.outcc1 = outconv(128, num_classes)
        self.up1 = up(256, 128)
        self.outcc2 = outconv(128, num_classes)
        self.up2 = up(384, 128)
        self.outcc3 = outconv(128, num_classes)
        self.up3 = up(384, 128)
        self.outcc4 = outconv(128, num_classes)
        self.up4 = up(384, 128)
        self.outcc = outconv(128, num_classes)
        self.tpp = TPPblock(128)
        self.weights = torch.nn.Parameter(torch.ones(6))

    def init_weights(self):
        pass

    def _clear_memory_buffer(self):
        pass
    
    def get_c2f_ensemble_output(self, output, weights):
    
        ensemble_prob = F.softmax(output[0], dim=1) * weights[0] / sum(weights)

        for i, outp_ele in enumerate(output[1]):
            upped_logit = F.upsample(outp_ele, size=output[0].shape[-1], mode='linear', align_corners=True)
            ensemble_prob = ensemble_prob + F.softmax(upped_logit, dim=1) * weights[i + 1] / sum(weights)
            
        ensemble_prob = ensemble_prob.unsqueeze(0)
        # ensemble_prob = torch.log(ensemble_prob + 1e-10).unsqueeze(0)
        return ensemble_prob

    def forward(self, x, mask):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x7 = self.down6(x6)
        # x7 = self.dac(x7)
        x7 = self.tpp(x7)
        x = self.up(x7, x6)
        y1 = self.outcc0(F.relu(x))
        # print("y1.shape=", y1.shape)
        x = self.up0(x, x5)
        y2 = self.outcc1(F.relu(x))
        # print("y2.shape=", y2.shape)
        x = self.up1(x, x4)
        y3 = self.outcc2(F.relu(x))
        # print("y3.shape=", y3.shape)
        x = self.up2(x, x3)
        y4 = self.outcc3(F.relu(x))
        # print("y4.shape=", y4.shape)
        x = self.up3(x, x2)
        y5 = self.outcc4(F.relu(x))
        # print("y5.shape=", y5.shape)
        x = self.up4(x, x1)
        y = self.outcc(x)
        # print("y.shape=", y.shape)

        outputs = self.get_c2f_ensemble_output((y, [y5, y4, y3, y2, y1], x), weights=self.ensem_weights)

        outputs = F.interpolate(
            input=outputs,
            scale_factor=[1, self.sample_rate],
            mode="nearest")

        return outputs