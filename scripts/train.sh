###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-03-15 12:39:18
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

# mstcn 1538574472
# asformer 19980125
# export DECORD_EOF_RETRY_MAX=20480
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer_rl_gtea.py --seed 400
