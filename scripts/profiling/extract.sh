
###
 # @Author       : Thyssen Wen
 # @Date         : 2022-10-24 16:10:45
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-11-03 11:34:15
 # @Description  : feature extract script
 # @FilePath     : /SVTAS/scripts/profiling/extract.sh
### 

export CUDA_VISIBLE_DEVICES=0

python tools/launch.py --mode extract -c config/extract/extract_feature/svtas_rl_50salads.py
