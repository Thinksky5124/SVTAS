'''
Author       : Thyssen Wen
Date         : 2023-10-18 20:25:40
LastEditors  : Thyssen Wen
LastEditTime : 2023-10-18 20:25:45
Description  : file content
FilePath     : /SVTAS/config/_base_/engine/profile_engine.py
'''
ENGINE = dict(
    name = "TorchStandaloneProfilerEngine",
    record = dict(
        name = "ValueRecord"
    ),
    iter_method = dict(
        name = "IterMethod",
        iter_num = 50,
        batch_size = 1,
        test_interval = 1,
        criterion_metric_name = "F1@0.50"
    ),
    checkpointor = dict(
        name = "TorchCheckpointor"
    )
)