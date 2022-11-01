###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-01 14:59:29
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python tools/launch.py --mode train --validate -c config/tas/rgb/mobilev2_tsm_3dtcn_breakfast.py --seed 0