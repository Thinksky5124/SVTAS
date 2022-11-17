###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-16 21:35:58
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

# mstcn 1538574472
# asformer 19980125
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/feature/asformer_gtea.py --seed 0
