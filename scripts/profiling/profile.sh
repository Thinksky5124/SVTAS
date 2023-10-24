###
 # @Author       : Thyssen Wen
 # @Date         : 2023-02-22 15:29:28
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-18 20:41:35
 # @Description  : file content
 # @FilePath     : /SVTAS/scripts/profile.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python tools/launch.py --mode profile -c config/profiling/svtas_rl_profiling.py