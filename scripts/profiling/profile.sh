###
 # @Author       : Thyssen Wen
 # @Date         : 2023-02-22 15:29:28
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-12-19 15:56:41
 # @Description  : file content
 # @FilePath     : /SVTAS/scripts/profiling/profile.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
# python tools/launch.py --mode profile -c config/profiling/svtas_rl_profiling.py
python tools/launch.py --mode profile -c config/profiling/svtas_rl_profiling_numerical_range.py