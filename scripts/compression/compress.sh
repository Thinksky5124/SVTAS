###
 # @Author       : Thyssen Wen
 # @Date         : 2023-12-13 15:32:05
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-12-13 15:32:13
 # @Description  : file content
 # @FilePath     : /SVTAS/scripts/compression/compress.sh
### 
export CUDA_VISIBLE_DEVICES=0
export DECORD_EOF_RETRY_MAX=20480

### gtea ###
python tools/launch.py --mode train -c config/compression/svtas-rl_gtea_deepspeed.py --seed 0