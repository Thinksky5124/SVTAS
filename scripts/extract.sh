
###
 # @Author       : Thyssen Wen
 # @Date         : 2022-10-24 16:10:45
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-04-25 20:55:14
 # @Description  : feature extract script
 # @FilePath     : /SVTAS/scripts/extract.sh
### 

export CUDA_VISIBLE_DEVICES=0

python tools/launch.py --mode extract -c config/extract/extract_feature/swin_transformer_3d_gtea.py
