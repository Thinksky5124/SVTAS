###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-15 10:12:35
 # @Description  : Test script
 # @FilePath     : /ETESVS/script/test.sh
### 

export CUDA_VISIBLE_DEVICES=0
python main.py  --test -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --weights=output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl