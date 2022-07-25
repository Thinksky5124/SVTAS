###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-23 09:53:42
 # @Description  : Test script
 # @FilePath     : /ETESVS/script/test.sh
### 

export CUDA_VISIBLE_DEVICES=1

#### GTEA ####
# python main.py  --test -c config/gtea/transeger/transeger_split1.yaml --weights=output/Transeger_gtea_split1/Transeger_gtea_split1_best.pkl
# python main.py  --test -c config/gtea/I3D_mstcn/i3d_mstcn_split1.yaml --weights=output/I3D_MSTCN_gtea_split1/I3D_MSTCN_gtea_split1_best.pkl
# python main.py  --test -c config/gtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --weights=output/MobileNetV2TSM_Memory_TCN_gtea_split1/MobileNetV2TSM_Memory_TCN_gtea_split1_best.pkl

#### 50Salads ####
# python main.py  --test -c config/50salads/transeger/transeger_split5.yaml --weights=output/Transeger_50salads_split5/Transeger_50salads_split5_best.pkl
# python main.py  --test -c config/50salads/I3D_mstcn/i3d_mstcn_split5.yaml --weights=output/I3D_MSTCN_50salads_split5/I3D_MSTCN_50salads_split5_best.pkl
# python main.py  --test -c config/50salads/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split5.yaml --weights=output/MobileNetV2TSM_Memory_TCN_50salads_split5/MobileNetV2TSM_Memory_TCN_50salads_split5_best.pkl

#### EGTEA ####
# python main.py  --test -c config/egtea/I3D_mstcn/i3d_mstcn_split1.yaml --weights=output/I3D_MSTCN_egtea_split1/I3D_MSTCN_egtea_split1_best.pkl
# python main.py  --test -c config/egtea/transeger/transeger_split3.yaml --weights=output/Transeger_egtea_split3/Transeger_egtea_split3_best.pkl
python main.py  --test -c config/egtea/TSM_memory_tcn/mobinetv2tsm_memory_tcn_split1.yaml --weights=output/MobileNetV2TSM_Memory_TCN_egtea_split1/MobileNetV2TSM_Memory_TCN_egtea_split1_best.pkl
