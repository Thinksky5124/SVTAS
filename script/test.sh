###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-16 09:52:12
 # @Description  : Test script
 # @FilePath     : /ETESVS/script/test.sh
### 

export CUDA_VISIBLE_DEVICES=0

#### 50SALADS ####
python main.py  --test -c config/50salads/transeger/transeger_split1.yaml --weights=output/Transeger_50salads_split1/Transeger_50salads_split1_best.pkl
python main.py  --test -c config/50salads/I3D_mstcn/i3d_mstcn_split3.yaml --weights=output/I3D_MSTCN_50salads_split3/I3D_MSTCN_50salads_split3_best.pkl
python main.py  --test -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --weights=output/MobileNetV2TSM_Memory_TCN_50salads_split1/MobileNetV2TSM_Memory_TCN_50salads_split1_best.pkl