###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-04-13 16:29:07
 # @Description  : Test script
 # @FilePath     : /SVTAS/scripts/test.sh
### 

export CUDA_VISIBLE_DEVICES=1

python tools/launch.py  --mode test -c config/svtas/rgb/swin_transformer_3d_small_brt_gtea.py --weights=output/SwinTransformer3D_BRT_64x2_gtea_split4_best.pt
