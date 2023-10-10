'''
Author       : Thyssen Wen
Date         : 2023-10-07 18:56:21
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 18:56:47
Description  : file content
FilePath     : /SVTAS/config/_base_/model_pipline/torch_model_pipline.py
'''
MODEL_PIPLINE = dict(
    name = "TorchModelPipline",
    grad_accumulate = dict(
        name = "GradAccumulate",
        accumulate_type = "conf"
    ),
    model = None,
    post_processing = None,
    criterion = None,
    optimizer = dict(
        name = "AdamWOptimizer",
        learning_rate = 0.0005,
        weight_decay = 1e-4,
        betas = (0.9, 0.999),
        finetuning_scale_factor=0.5,
        no_decay_key = [],
        finetuning_key = [],
        freeze_key = [],
    ),
    lr_scheduler = dict(
        name = "MultiStepLR",
        step_size = [50],
        gamma = 0.1,
    )
)