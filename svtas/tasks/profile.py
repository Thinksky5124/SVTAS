'''
Author: Thyssen Wen
Date: 2022-03-17 12:12:57
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-05 15:25:23
Description: test script api
FilePath     : /SVTAS/svtas/tasks/profile.py
'''
import torch
import torch.profiler
from ..utils.logger import get_logger
import time
import numpy as np
import datetime

from mmcv.cnn.utils.flops_counter import get_model_complexity_info
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import clever_format
from ..utils.collect_env import collect_env

@torch.no_grad()
def profile(local_rank,
            nprocs,
            cfg,
            args,
            use_amp=False,
            weights=None):
    logger = get_logger("SVTAS")
    if hasattr(cfg,'PROFILER'):
        wait = cfg.PROFILER.get('wait', 1)
        warmup = cfg.PROFILER.get('warmup', 1)
        active = cfg.PROFILER.get('active', 3)
        repeat = cfg.PROFILER.get('repeat', 2)
        record_shapes = cfg.PROFILER.get('record_shapes', True)
        profile_memory = cfg.PROFILER.get('profile_memory', True)
        with_stack = cfg.PROFILER.get('with_stack', True)
    else:
        wait = 1
        warmup = 1
        active = 3
        repeat = 2
        record_shapes = True
        profile_memory = True
        with_stack = True

    assert local_rank <= 0, "Profiler only support single GPU and computer noew!"
    profiler_logger_path = f"./"+ "output" + f"/{cfg.model_name}" +"/model_profile_" + datetime.datetime.now().strftime("%Y%m%d%H%M%S") + ".log"
    
    if args.use_tensorboard and local_rank <= 0:
        tensorboard_writer = get_logger("SVTAS", tensorboard=args.use_tensorboard)
    
    # env info logger
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # 1. Construct model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 1.construct model
    model = build_model(cfg.MODEL).to(device)
    criterion = build_loss(cfg.MODEL.loss).to(device)

    # 2. Construct solver.
    optimizer_cfg = cfg.OPTIMIZER
    optimizer_cfg['model'] = model
    optimizer = build_optimizer(optimizer_cfg)
    # grad to zeros
    optimizer.zero_grad()


    # 2. Construct dataset and dataloader.
    # default num worker: 0, which means no subprocess will be created
    num_workers = cfg.DATASET.get('num_workers', 0)
    test_num_workers = cfg.DATASET.get('test_num_workers', num_workers)
    temporal_clip_batch_size = cfg.DATASET.get('temporal_clip_batch_size', 3)
    video_batch_size = cfg.DATASET.get('video_batch_size', 8)
    sliding_concate_fn = build_pipline(cfg.COLLATE.test)
    test_Pipeline = build_pipline(cfg.DATASETPIPLINE.test)
    test_dataset_config = cfg.DATASET.test
    test_dataset_config['pipeline'] = test_Pipeline
    test_dataset_config['temporal_clip_batch_size'] = temporal_clip_batch_size
    test_dataset_config['video_batch_size'] = video_batch_size * nprocs
    test_dataset_config['local_rank'] = local_rank
    test_dataset_config['nprocs'] = nprocs
    test_dataloader = torch.utils.data.DataLoader(
        build_dataset(test_dataset_config),
        batch_size=temporal_clip_batch_size,
        num_workers=test_num_workers,
        collate_fn=sliding_concate_fn)
    
    if weights is not None:
        checkpoint = torch.load(weights)

        state_dicts = checkpoint['model_state_dict']

        if nprocs > 1:
            model.module.load_state_dict(state_dicts)
        else:
            model.load_state_dict(state_dicts)

    def model_forward(data_dict):
        # move data
        input_data = {}
        for key, value in data_dict.items():
            if torch.is_tensor(value):
                if torch.cuda.is_available():
                    input_data[key] = value.cuda()
                else:
                    input_data[key] = value

        outputs = model(input_data)
        loss_dict = criterion(outputs, input_data)
        optimizer.zero_grad()

        loss = loss_dict["loss"]
        loss.backward()
        optimizer.step()

        score = outputs['output']
        return score, loss_dict
    
    clip_seg_num = list(cfg.DATASETPIPLINE.test.sample.get('clip_seg_num_dict', {'sample':32}).values())[0]
    sample_rate = list(cfg.DATASETPIPLINE.test.sample.get('sample_rate_dict', {'sample':1}).values())[0]
    # model param flops caculate
    if cfg.MODEL.architecture not in ["FeatureSegmentation"]:
        for transform_op in list(cfg.DATASETPIPLINE.test.transform.transform_dict.values())[0]:
            if list(transform_op.keys())[0] in ['CenterCrop']:
                image_size = transform_op['CenterCrop']['size']
        x_shape = [clip_seg_num, 3, image_size, image_size]
        mask_shape = [clip_seg_num * sample_rate]
        labels_shape = [clip_seg_num * sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).to(device)
            mask = torch.randn([optimal_batch_size] + mask_shape).to(device)
            label = torch.ones([optimal_batch_size] + labels_shape).to(device)
            return dict(input_data=dict(imgs=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)
    else:
        x_shape = [clip_seg_num, 2048]
        mask_shape = [clip_seg_num * sample_rate]
        labels_shape = [clip_seg_num * sample_rate]
        input_shape = (x_shape, mask_shape, labels_shape)
        def input_constructor(input_shape, optimal_batch_size=1):
            x_shape, mask_shape, labels_shape = input_shape
            x = torch.randn([optimal_batch_size] + x_shape).to(device)
            mask = torch.randn([optimal_batch_size] + mask_shape).to(device)
            label = torch.ones([optimal_batch_size] + labels_shape).to(device)
            return dict(input_data=dict(feature=x, masks=mask, labels=label))
        dummy_input = input_constructor(input_shape)
    # print(model)
    # tensorboard_writer.add_graph(model, input_to_model=[x, mask, torch.ones(1).cuda()])

    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(profiler_logger_path),
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack
        ) as prof:
            model.train()
            with torch.enable_grad():
                for step, data in enumerate(test_dataloader):
                    if step >= (wait+warmup+active)*repeat:
                        break
                    batch_data = data[0]
                    model_forward(batch_data)
                    prof.step()  # Need to call this at the end of each step to notify profiler of steps' boundary.

    model.eval()
    # mmcv caculate param and flops
    logger.info("="*20)
    logger.info('Use mmcv get_model_complexity_info function')
    flops_number, params_number = get_model_complexity_info(model, input_shape=input_shape, input_constructor=input_constructor, print_per_layer_stat=False, as_strings=False)
    flops_per_image_number = flops_number / clip_seg_num
    flops, params = clever_format([flops_number, params_number], "%.6f")
    flops_per_image, params = clever_format([flops_per_image_number, params_number], "%.6f")
    logger.info("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=clip_seg_num))
    logger.info("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops + ", Total params:" + params)
    logger.info(f"Computed strength is {flops_number/params_number} FLOPS/Byte.")
    logger.info("="*20)

    # fvcore caculate param and flops
    logger.info('Use fvcore FlopCountAnalysis function')
    inputs = (dummy_input['input_data'])
    flops = FlopCountAnalysis(model, inputs)
    logger.info("flop_count_table: \n" + flop_count_table(flops))
    flops_number = flops.total()
    flops_per_image_number = flops_number / clip_seg_num
    flops = clever_format([flops_number], "%.6f")
    flops_per_image = clever_format([flops_per_image_number], "%.6f")
    logger.info("Hitp: This FLOPs is caculation by {clip_seg_num:d} imgs".format(clip_seg_num=clip_seg_num))
    logger.info("Per Image FLOPs:"+ flops_per_image + ", Total FLOPs:" + flops)
    logger.info("="*20)

    # model fps caculate
    dummy_input = dummy_input['input_data']
    logger.info('Caculate model fps (single frame test times)')
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = repeat
    timings = np.zeros((repetitions, 1))

    #GPU-WARM-UP
    for _ in range(10):
        _ = model(dummy_input)

    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    mean_fps = 1000. / mean_syn * clip_seg_num
    logger.info('Mean@1 {mean_syn:.3f}ms, Std@5 {std_syn:.3f}ms, FPS@1 {mean_fps:.2f}'.format(mean_syn=mean_syn, std_syn=std_syn, mean_fps=mean_fps))
    logger.info('Model single forward test time(ms) {mean_syn:.3f}ms'.format(mean_syn=mean_syn))
    logger.info("="*20)

    # model latency time
    logger.info('Caculate model Throughput')
    repetitions=repeat
    total_time = 0
    # it should be modify by every model
    optimal_batch_size=1
    dummy_input = input_constructor(input_shape, optimal_batch_size=optimal_batch_size)['input_data']
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender) / 1000
            total_time += curr_time
    Throughput = (repetitions * optimal_batch_size) / total_time
    logger.info("Final Throughput: {Throughput:.2f} V/s, Measuring by batch_size: {Batch_size:d}".format(Throughput=Throughput, Batch_size=optimal_batch_size))
    logger.info("="*20)