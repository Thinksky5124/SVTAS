###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-05 21:48:09
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/rgb/i3d_rgb_flow_asformer_gtea.py --seed 1538574472