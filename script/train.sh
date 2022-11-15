###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-14 21:32:39
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/rgb/x3d_rgb_flow_conformer_gtea.py --seed 0
