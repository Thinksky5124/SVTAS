'''
Author       : Thyssen Wen
Date         : 2023-10-15 16:02:31
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-15 16:02:34
Description  : file content
FilePath     : /SVTAS/config/_base_/engine/deepspeed_engine.py
'''
ENGINE = dict(
    name = "DeepSpeedDistributedDataParallelEngine",
    record = dict(
        name = "ValueRecord"
    ),
    iter_method = dict(
        name = "EpochMethod",
        epoch_num = 80,
        batch_size = 1,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "DeepSpeedCheckpointor"
    )
)
