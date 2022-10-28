###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-21 23:42:16
 # @Description  : Infer script
 # @FilePath     : /SVTAS/script/infer.sh
### 

# Debug Infer Model
# python tools/launch.py --mode infer --validate -c config/infer/mobinetv2tsm_memory_tcn_split1.yaml -w output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl

# Only Infer
# python tools/launch.py --mode infer -c config/infer/mobinetv2tsm_memory_tcn_split1.yaml -w output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl

# export Infer model
# python tools/infer/export_model_to_onnx.py -c config/infer/mobinetv2tsm_3Dtcn_split1.yaml -w output/MobileNetV2TSM_3D_TCN_gtea_split1/MobileNetV2TSM_3D_TCN_gtea_split1_best.pkl

# deployment of infer
python tools/infer/infer.py -m output/infer/MobileNetV2TSM_3D_TCN_gtea_split1/MobileNetV2TSM_3D_TCN_gtea_split1.onnx -i data/gtea/Videos/S1_Cheese_C1.mp4 -l data/gtea/mapping.txt -o output/output.mp4