###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-03-30 09:24:03
 # @Description  : Test script
 # @FilePath     : /SVTAS/scripts/test.sh
### 

export CUDA_VISIBLE_DEVICES=1

python tools/launch.py  --mode test -c config/svtas/rgb/swin_transformer_3d_small_sbp_fc_breakfast.py --weights=output/SwinTransformer3DSBP_FC_128x4_breakfast_split1/SwinTransformer3DSBP_FC_128x4_breakfast_split1_best.pt
