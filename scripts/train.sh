###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-03-03 19:12:27
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

# mstcn 1538574472
# asformer 19980125
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer_gtea.py --seed 400
# python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer_50salads.py --seed 400
# python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer_breakfast.py --seed 400
