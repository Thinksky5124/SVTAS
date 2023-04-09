###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-04-09 20:40:35
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

# mstcn 1538574472
# asformer 19980125
export DECORD_EOF_RETRY_MAX=20480
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer_rl_breakfast.py --seed 0
