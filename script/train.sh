###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-30 19:18:59
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

### gtea ###
python tools/launch.py --mode train --validate -c config/tas/rgb/action_clip_ms_tcn_gtea.py --seed 1538574472