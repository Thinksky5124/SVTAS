###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors: Thinksky5124 Thinksky5124@outlook.com
 # @LastEditTime: 2023-05-04 10:48:07
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

# mstcn 1538574472
# asformer 19980125
export DECORD_EOF_RETRY_MAX=20480
### gtea ###
# python tools/launch.py --mode train --validate -c config/svtas/feature/fc_50salads.py --seed 0 # 记得检查一下
# python tools/launch.py --mode train --validate -c config/svtas/feature/asformer_50salads.py --seed 0 # 记得检查一下
python tools/launch.py --mode train --validate -c /home/wenwujun/SVTAS/config/svtas/final/re_swin_transformer_3d_base_brt_breakfast_2.py --seed 0