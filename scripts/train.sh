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
python tools/launch.py --mode train -c config/svtas/svtas-rl/svtas_rl_gtea.py --seed 0