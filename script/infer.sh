###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-09-24 20:55:54
 # @Description  : Infer script
 # @FilePath     : /ETESVS/script/infer.sh
### 

# Debug Infer Model
python main.py --mode infer --validate -c config/infer/mobinetv2tsm_memory_tcn_split1.yaml -w output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl
# Only Infer
# python main.py --mode infer -c config/infer/mobinetv2tsm_memory_tcn_split1.yaml -w output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl