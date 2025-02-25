'''
Author: Thyssen Wen
Date: 2022-04-27 16:24:59
LastEditors  : Thyssen Wen
LastEditTime : 2022-07-12 15:37:30
Description: recorder construct function
FilePath     : /ETESVS/utils/recorder.py
'''
from utils.logger import AverageMeter

def build_recod(architecture_type, mode):
    assert mode in ["train", "validation", "test"]
    if architecture_type in ["StreamSegmentation2DWithNeck"]:
        if mode == "train":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'loss': AverageMeter('loss', '7.5f'),
                    'lr': AverageMeter('lr', 'f', need_avg=False),
                    'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                    'Acc': AverageMeter("Acc", '.5f'),
                    'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                    'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                    'neck_loss': AverageMeter("neck_loss", '.5f'),
                    'head_loss': AverageMeter("head_loss", '.5f')
                    }
        elif mode == "validation":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                   'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                   'neck_loss': AverageMeter("neck_loss", '.5f'),
                   'head_loss': AverageMeter("head_loss", '.5f')
                  }
    elif architecture_type in ["StreamSegmentation2DWithBackbone", "MulModStreamSegmentation",
                "StreamSegmentation3DWithBackbone"]:
        if mode == "train":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'loss': AverageMeter('loss', '7.5f'),
                    'lr': AverageMeter('lr', 'f', need_avg=False),
                    'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                    'Acc': AverageMeter("Acc", '.5f'),
                    'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                    'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                    'head_loss': AverageMeter("head_loss", '.5f')
                    }
        elif mode == "validation":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                   'backbone_loss': AverageMeter("backbone_loss", '.5f'),
                   'head_loss': AverageMeter("head_loss", '.5f')
                  }
    elif architecture_type in ["FeatureSegmentation", "Recognition2D", "Recognition3D",
                                "StreamSegmentation3D", "StreamSegmentation2D", "Transeger"]:
        if mode == "train":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'loss': AverageMeter('loss', '7.5f'),
                    'lr': AverageMeter('lr', 'f', need_avg=False),
                    'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                    'Acc': AverageMeter("Acc", '.5f'),
                    'Seg_Acc': AverageMeter("Seg_Acc", '.5f')
                    }
        elif mode == "validation":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f')
                  }
    elif architecture_type in ['StreamSegmentation2DMultiLabel']:
        if mode == "train":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'action_loss': AverageMeter('action_loss', '7.5f'),
                    'branch_loss': AverageMeter('branch_loss', '7.5f'),
                    'lr': AverageMeter('lr', 'f', need_avg=False),
                    'F1Action@0.5': AverageMeter("F1Action@0.5", '.5f'),
                    'F1Branch@0.5': AverageMeter("F1Branch@0.5", '.5f'),
                    'ActionAcc': AverageMeter("Acc", '.5f'),
                    'ActionSeg_Acc': AverageMeter("ActionSeg_Acc", '.5f'),
                    'BranchAcc': AverageMeter("Acc", '.5f'),
                    'BranchSeg_Acc': AverageMeter("BranchSeg_Acc", '.5f')
                    }
        elif mode == "validation":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'action_loss': AverageMeter('action_loss', '7.5f'),
                    'branch_loss': AverageMeter('branch_loss', '7.5f'),
                    'F1Action@0.5': AverageMeter("F1Action@0.5", '.5f'),
                    'F1Branch@0.5': AverageMeter("F1Branch@0.5", '.5f'),
                    'ActionAcc': AverageMeter("Acc", '.5f'),
                    'ActionSeg_Acc': AverageMeter("ActionSeg_Acc", '.5f'),
                    'BranchAcc': AverageMeter("Acc", '.5f'),
                    'BranchSeg_Acc': AverageMeter("BranchSeg_Acc", '.5f')
                    }
    elif architecture_type in ["SegmentationCLIP"]:
        if mode == "train":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                    'reader_time': AverageMeter('reader_time', '.5f'),
                    'loss': AverageMeter('loss', '7.5f'),
                    'lr': AverageMeter('lr', 'f', need_avg=False),
                    'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                    'Acc': AverageMeter("Acc", '.5f'),
                    'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                    'img_seg_loss': AverageMeter("img_seg_loss", '.5f'),
                    'clip_loss': AverageMeter("clip_loss", '.5f'),
                    }
        elif mode == "validation":
            return {'batch_time': AverageMeter('batch_cost', '.5f'),
                   'reader_time': AverageMeter('reader_time', '.5f'),
                   'loss': AverageMeter('loss', '7.5f'),
                   'F1@0.5': AverageMeter("F1@0.50", '.5f'),
                   'Acc': AverageMeter("Acc", '.5f'),
                   'Seg_Acc': AverageMeter("Seg_Acc", '.5f'),
                   'img_seg_loss': AverageMeter("img_seg_loss", '.5f'),
                   'clip_loss': AverageMeter("clip_loss", '.5f'),
                  }
    else:
        raise NotImplementedError