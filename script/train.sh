###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-08 17:17:46
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/feature/conformer_gtea.py --seed 1538574472