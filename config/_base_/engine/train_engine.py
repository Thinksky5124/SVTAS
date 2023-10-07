'''
Author       : Thyssen Wen
Date         : 2023-10-07 18:51:35
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-07 18:53:09
Description  : file content
FilePath     : /SVTAS/config/_base_/engine/train_engine.py
'''
ENGINE = dict(
    name = "BaseImplementEngine",
    record = dict(
        name = "StreamValueRecord"
    ),
    iter_method = dict(
        name = "StreamEpochMethod",
        epoch_num = 50,
        batch_size = 1,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)