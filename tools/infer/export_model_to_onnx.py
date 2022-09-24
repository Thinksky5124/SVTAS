'''
Author       : Thyssen Wen
Date         : 2022-09-03 15:05:29
LastEditors  : Thyssen Wen
LastEditTime : 2022-09-24 13:11:53
Description  : Export torch model to ONNX
FilePath     : \ETESVS\tools\infer\export_model_to_onnx.py
'''
import argparse
import os
import sys
path = os.path.join(os.getcwd())
sys.path.append(path)
import model.builder as model_builder
from utils.logger import get_logger
import torch
import numpy as np
import onnx
import onnxruntime

from utils.config import get_config

@torch.no_grad()
def export_model_to_onnx(cfg,
                         args):
    logger = get_logger("SVTAS")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # construct torch model
    model = model_builder.build_model(cfg.MODEL).to(device)
    checkpoint = torch.load(args.weights)
    state_dicts = checkpoint['model_state_dict']
    model.load_state_dict(state_dicts)

    # export path construct
    export_path = os.path.join(args.export_path, cfg.model_name, cfg.model_name + ".onnx")

    if os.path.exists(os.path.join(args.export_path, cfg.model_name)) is False:
        os.mkdir(os.path.join(args.export_path, cfg.model_name))

    # model param flops caculate
    if cfg.MODEL.architecture not in ["FeatureSegmentation"]:
        x_shape = [cfg.DATASET.test.clip_seg_num, 3, 224, 224]
        mask_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        labels_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).cuda()
            mask = torch.randn([optimal_batch_size] + mask_shape).cuda()
            label = torch.ones([optimal_batch_size] + labels_shape).cuda()
            return dict(input_data=dict(imgs=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)
    else:
        x_shape = [cfg.DATASET.test.clip_seg_num, 2048]
        mask_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        labels_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).cuda()
            mask = torch.randn([optimal_batch_size] + mask_shape).cuda()
            label = torch.ones([optimal_batch_size] + labels_shape).cuda()
            return dict(input_data=dict(feature=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)

    logger.info("Start exporting ONNX model!")
    torch.onnx.export(
        model,
        dummy_input['input_data'],
        export_path,
        opset_version=11,
        input_names=['input_data', 'masks'],
        output_names=['output'])
    logger.info("Finish exporting ONNX model to " + export_path + " !")
    
    # Model check
    onnx_model = onnx.load(export_path)
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        logger.info("UnPass ONNX checker checking ONNX model!")
    else:
        logger.info("Pass ONNX checker checking ONNX model!")

    # precision alignment
    # onnx
    ort_session = onnxruntime.InferenceSession(export_path)
    if cfg.MODEL.architecture not in ["FeatureSegmentation"]:
        ort_inputs = {'input_data': dummy_input['input_data']['imgs'].cpu().numpy(), 'masks': dummy_input['input_data']['masks'].cpu().numpy()}
    else:
        ort_inputs = {'input_data': dummy_input['input_data']['feature'].cpu().numpy(), 'masks': dummy_input['input_data']['masks'].cpu().numpy()}
    ort_output = ort_session.run(None, ort_inputs)
    # torch
    torch_output = model(dummy_input['input_data']).cpu().numpy()

    precision_judge = np.allclose(ort_output, torch_output)

    if precision_judge:
        logger.info('Pass Precision Alignment!')
    else:
        logger.info('UnPass Precision Alignment!')

def parse_args():
    parser = argparse.ArgumentParser("SVTAS export model script")
    parser.add_argument('-c',
                        '--config',
                        type=str,
                        default='configs/example.yaml',
                        help='config file path')
    parser.add_argument('--export_path',
                        type=str,
                        default='output/infer/',
                        help='where export a model')
    parser.add_argument('-o',
                        '--override',
                        action='append',
                        default=[],
                        help='config options to be overridden')
    parser.add_argument('-w',
                        '--weights',
                        type=str,
                        help='weights for finetuning or testing')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(0)
    return args


def main():
    args = parse_args()
    cfg = get_config(args.config, overrides=args.override, tensorboard=False, logger_path = "output/infer")

    export_model_to_onnx(cfg, args)

if __name__ == '__main__':
    main()
