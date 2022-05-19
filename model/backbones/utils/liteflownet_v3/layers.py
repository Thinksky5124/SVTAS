'''
Author       : Thyssen Wen
Date         : 2022-05-19 14:36:20
LastEditors  : Thyssen Wen
LastEditTime : 2022-05-19 14:55:11
Description  : LiteFlowNet V3 layers define
FilePath     : /ETESVS/model/backbones/utils/liteflownet_v3/layers.py
'''
import torch
import torch.nn as nn
from .correlation_package.correlation import Correlation

class BackWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.backwarp_tenGrid = {}

    def forward(self, tenInput, tenFlow):
        if str(tenFlow.shape) not in self.backwarp_tenGrid:
            tenHor = torch.linspace(-1.0 + (1.0 / tenFlow.shape[3]), 1.0 - (1.0 / tenFlow.shape[3]), tenFlow.shape[3]).view(1, 1, 1, -1).expand(-1, -1, tenFlow.shape[2], -1)
            tenVer = torch.linspace(-1.0 + (1.0 / tenFlow.shape[2]), 1.0 - (1.0 / tenFlow.shape[2]), tenFlow.shape[2]).view(1, 1, -1, 1).expand(-1, -1, -1, tenFlow.shape[3])

            self.backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

        tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)

        return nn.functional.grid_sample(input=tenInput, grid=(self.backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=False)

class Features(nn.Module):
    def __init__(self):
        super(Features, self).__init__()

        self.netOne = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=1, padding=3),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netTwo = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netThr = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFou = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netFiv = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        self.netSix = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

    def forward(self, tenInput):
        tenOne = self.netOne(tenInput)
        tenTwo = self.netTwo(tenOne)
        tenThr = self.netThr(tenTwo)
        tenFou = self.netFou(tenThr)
        tenFiv = self.netFiv(tenFou)
        tenSix = self.netSix(tenFiv)

        return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]

class Matching(nn.Module):
    def __init__(self, intLevel):
        super(Matching, self).__init__()

        self.fltBackwarp = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]
        
        self.crossCorr = Correlation(pad_size=4, kernel_size=1, max_displacement=4, stride1=1, stride2=1)

        if intLevel == 4:
            self.autoCorr = Correlation(pad_size=6, kernel_size=1, max_displacement=6, stride1=1, stride2=2)
        elif intLevel == 3:
            self.autoCorr = Correlation(pad_size=8, kernel_size=1, max_displacement=8, stride1=1, stride2=2)

        if intLevel > 4:
            self.confFeat = None
            self.corrFeat = None
        
        if intLevel <= 4:
            self.confFeat = nn.Sequential(
                nn.Conv2d(in_channels=[0, 0, 0, 1 + 81, 1 + 49][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )
            self.dispNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=2, kernel_size=5, stride=1, padding=2)
                )
            self.confNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=5, stride=1, padding=2),
                nn.Sigmoid()
                )

            self.corrFeat = nn.Sequential(
                nn.Conv2d(in_channels=[0, 0, 0, 64 + 81 + 1, 96 + 81 + 1][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1))
            self.corrScalar = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
            )
            self.corrOffset = nn.Sequential(
                nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=False, negative_slope=0.1),
                nn.Conv2d(in_channels=32, out_channels=81, kernel_size=1, stride=1, padding=0)
            )

        if intLevel == 6:
            self.netUpflow = None

        elif intLevel != 6:
            self.netUpflow = nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1, bias=False, groups=2)

        if intLevel == 4 or intLevel == 3:
            self.netUpconf = nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=4, stride=2, padding=1, bias=False, groups=1)

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=81, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
        )

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow, tenConf, backwarp):
        if self.confFeat:
            tenConf = self.netUpconf(tenConf)
            tenCorrelation = nn.functional.leaky_relu(input=self.autoCorr(tenFeaturesFirst, tenFeaturesFirst), negative_slope=0.1, inplace=False)
            confFeat = self.confFeat(torch.cat([tenCorrelation, tenConf], 1))
            tenConf = self.confNet(confFeat)
            tenDisp = self.dispNet(confFeat)
            
        if tenFlow is not None:
            tenFlow = self.netUpflow(tenFlow)
        if self.corrFeat:
            tenFlow = backwarp(tenInput=tenFlow, tenFlow=tenDisp)

        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackwarp)

        tenCorrelation = nn.functional.leaky_relu(input=self.crossCorr(tenFeaturesFirst, tenFeaturesSecond), negative_slope=0.1, inplace=False)

        if self.corrFeat:
            corrfeat = self.corrFeat(torch.cat([tenFeaturesFirst, tenCorrelation, tenConf], 1))
            corrscalar = self.corrScalar(corrfeat)
            corroffset = self.corrOffset(corrfeat)
            tenCorrelation = corrscalar * tenCorrelation + corroffset
        
        return (tenFlow if tenFlow is not None else 0.0) + self.netMain(tenCorrelation), tenConf

class Subpixel(nn.Module):
    def __init__(self, intLevel):
        super(Subpixel, self).__init__()

        self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 0, 130, 194, 258, 386 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=2, kernel_size=[ 0, 0, 0, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 0, 2, 2, 1, 1 ][intLevel])
        )

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow, backwarp):
        if tenFlow is not None:
            tenFeaturesSecond = backwarp(tenInput=tenFeaturesSecond, tenFlow=tenFlow * self.fltBackward)

        return (tenFlow if tenFlow is not None else 0.0) + self.netMain(torch.cat([ tenFeaturesFirst, tenFeaturesSecond, tenFlow ], 1))

class Regularization(nn.Module):
    def __init__(self, intLevel):
        super(Regularization, self).__init__()

        self.fltBackward = [ 0.0, 0.0, 0.0, 5.0, 2.5, 1.25, 0.625 ][intLevel]

        self.intUnfold = [ 0, 0, 7, 5, 5, 3, 3 ][intLevel]

        if intLevel > 4:
            self.netFeat = nn.Sequential()

        elif intLevel <= 4:
            self.netFeat = nn.Sequential(
                nn.Conv2d(in_channels=[ 0, 0, 32, 64, 96, 128, 192 ][intLevel], out_channels=128, kernel_size=1, stride=1, padding=0),
                nn.LeakyReLU(inplace=False, negative_slope=0.1)
            )

        self.netMain = nn.Sequential(
            nn.Conv2d(in_channels=[ 0, 0, 131, 131, 131, 131, 195 ][intLevel], out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=False, negative_slope=0.1)
        )

        if intLevel >= 5:
            self.netDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], kernel_size=[ 0, 0, 7, 5, 5, 3, 3 ][intLevel], stride=1, padding=[ 0, 0, 3, 2, 2, 1, 1 ][intLevel])
            )

        elif intLevel < 5:
            self.netDist = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], kernel_size=([ 0, 0, 0, 5, 5, 3, 3 ][intLevel], 1), stride=1, padding=([ 0, 0, 0, 2, 2, 1, 1 ][intLevel], 0)),
                nn.Conv2d(in_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], out_channels=[ 0, 0, 0, 25, 25, 9, 9 ][intLevel], kernel_size=(1, [ 0, 0, 0, 5, 5, 3, 3 ][intLevel]), stride=1, padding=(0, [ 0, 0, 0, 2, 2, 1, 1 ][intLevel]))
            )
        
        if intLevel == 5 or intLevel == 4:
            self.confNet = nn.Sequential(
                nn.Conv2d(in_channels=32, out_channels=1, kernel_size=[0, 0, 0, 0, 5, 3][intLevel], stride=1, padding=[0, 0, 0, 0, 2, 1][intLevel]),
                nn.Sigmoid()
            )
        else:
            self.confNet = None

        self.netScaleX = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)
        self.netScaleY = nn.Conv2d(in_channels=[ 0, 0, 49, 25, 25, 9, 9 ][intLevel], out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, tenFirst, tenSecond, tenFeaturesFirst, tenFeaturesSecond, tenFlow, backwarp):
        tenDifference = (tenFirst - backwarp(tenInput=tenSecond, tenFlow=tenFlow * self.fltBackward)).pow(2.0).sum(1, True).sqrt().detach()

        tenFeaturesFirst = self.netFeat(tenFeaturesFirst)

        mainfeat = self.netMain(torch.cat([ tenDifference, tenFlow - tenFlow.view(tenFlow.shape[0], 2, -1).mean(2, True).view(tenFlow.shape[0], 2, 1, 1), tenFeaturesFirst ], 1))
        tenDist = self.netDist(mainfeat)
        
        tenConf = None
        if self.confNet:
            tenConf = self.confNet(mainfeat)

        tenDist = tenDist.pow(2.0).neg()
        tenDist = (tenDist - tenDist.max(1, True)[0]).exp()

        tenDivisor = tenDist.sum(1, True).reciprocal()

        tenScaleX = self.netScaleX(tenDist * nn.functional.unfold(input=tenFlow[:, 0:1, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor
        tenScaleY = self.netScaleY(tenDist * nn.functional.unfold(input=tenFlow[:, 1:2, :, :], kernel_size=self.intUnfold, stride=1, padding=int((self.intUnfold - 1) / 2)).view_as(tenDist)) * tenDivisor

        return torch.cat([ tenScaleX, tenScaleY ], 1), tenConf