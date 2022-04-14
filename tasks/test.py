'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors: Thyssen Wen
LastEditTime: 2022-04-14 19:26:41
Description: test script api
FilePath: /ETESVS/tasks/test.py
'''
from typing import OrderedDict
import torch
from utils.logger import get_logger
from .runner import Runner

import model.builder as builder
from dataset.segmentation_dataset import SegmentationDataset
from utils.metric import SegmentationMetric
from dataset.pipline import Pipeline
from dataset.pipline import BatchCompose
from model.post_precessings import PostProcessing
from torchinfo import summary
from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from thop import clever_format

try:
    from apex import amp
    from apex.parallel import convert_syncbn_model
    from apex.parallel import DistributedDataParallel as DDP
except:
    pass

@torch.no_grad()
def test(cfg,
         args,
         local_rank,
         nprocs,
         use_amp=False,
         weights=None):
    logger = get_logger("ETESVS")
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("ETESVS", tensorboard=args.use_tensorboard)
    # wheather use amp
    if use_amp is True:
        logger.info("use amp")

    # 1. Construct model.
    if local_rank < 0:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 1.construct model
        model = builder.build_model(cfg.MODEL).cuda()

        # wheather to use amp
        if use_amp is True:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
        # 1.construct model
        model = builder.build_model(cfg.MODEL).cuda(local_rank)
    
        # wheather to use amp
        if use_amp is True:
            device = torch.device('cuda:{}'.format(local_rank))
            model = convert_syncbn_model(model).to(device)
            model = DDP(model, delay_allreduce=True)
        else:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    

    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = BatchCompose(**cfg.COLLATE)
    test_Pipeline = Pipeline(**cfg.PIPELINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_Pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader = torch.utils.data.DataLoader(
        SegmentationDataset(**test_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)

    if local_rank < 0:
        checkpoint = torch.load(weights)
    else:
        # configure map_location properly
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        checkpoint = torch.load(weights, map_location=map_location)

    state_dicts = checkpoint['model_state_dict']

    if nprocs > 1:
        state_dicts_new = OrderedDict()   # create new OrderedDict that does not contain `module.`
        for k, v in state_dicts.items():
            name = k.replace('module.', '')
            state_dicts_new[name] = v
        model.module.load_state_dict(state_dicts_new)
    else:
        state_dicts_new = state_dicts
        model.load_state_dict(state_dicts_new)

    if use_amp is True:
        amp.load_state_dict(checkpoint['amp'])

    # add params to metrics
    Metric = SegmentationMetric(**cfg.METRIC)

    post_processing = PostProcessing(
        num_classes=cfg.MODEL.head.num_classes,
        clip_seg_num=cfg.MODEL.neck.clip_seg_num,
        sliding_window=cfg.DATASET.test.sliding_window,
        sample_rate=cfg.DATASET.test.sample_rate)

    runner = Runner(logger=logger,
                video_batch_size=video_batch_size,
                Metric=Metric,
                cfg=cfg,
                model=model,
                post_processing=post_processing,
                use_amp=use_amp,
                nprocs=nprocs,
                local_rank=local_rank,
                runner_mode='test')

    runner.epoch_init()

    for i, data in enumerate(test_dataloader):
        runner.run_one_iter(data=data)

    if local_rank <= 0:
        # metric output
        runner.Metric.accumulate()

        # model param flops caculate
        x_shape = [cfg.MODEL.neck.clip_seg_num, 3, 244, 244]
        mask_shape = [cfg.DATASET.test.clip_seg_num * cfg.DATASET.test.sample_rate]
        input_shape = (x_shape, mask_shape)
        def input_constructor(input_shape):
            x_shape, mask_shape = input_shape
            x = torch.randn([1] + x_shape).cuda()
            mask = torch.randn([1] + mask_shape).cuda()
            idx = torch.randn([1] + [1]).cuda()
            return dict(imgs=x, masks=mask, idx=idx)
        output = input_constructor(input_shape)
        x, mask = output["imgs"], output["masks"]
        # tensorboard_writer.add_graph(model, input_to_model=[x, mask, torch.ones(1).cuda()])
        summary(model, input_size=[x.shape, mask.shape, [1]], col_names=["kernel_size", "output_size", "num_params", "mult_adds"])
        print("="*20)
        print('Use mmcv get_model_complexity_info function')
        flops_number, params_number = get_model_complexity_info(model, input_shape=input_shape, input_constructor=input_constructor, print_per_layer_stat=False, as_strings=False)
        flops_per_image_number = flops_number / cfg.DATASET.test.clip_seg_num
        flops, params = clever_format([flops_number, params_number], "%.3f")
        flops_per_image, params = clever_format([flops_per_image_number, params_number], "%.3f")
        print("Hitp: This FLOPs is caculation by", cfg.DATASET.test.clip_seg_num, "imgs")
        print("Per Image FLOPs:", flops_per_image, ", Total FLOPs:", flops, ", Total params", params)
        print("="*20)