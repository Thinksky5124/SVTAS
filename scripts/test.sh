###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-04-25 22:20:51
 # @Description  : Test script
 # @FilePath     : /SVTAS/scripts/test.sh
### 

export CUDA_VISIBLE_DEVICES=1

python tools/launch.py  --mode test -c config/svtas/feature/block_recurrent_transformer_rl_50salads.py --weights=output/Stream_BRT_128x8_50salads_split1/Stream_BRT_128x8_50salads_split1_best.pt
