###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-31 16:46:00
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

### gtea ###
python tools/launch.py --mode train --validate -c config/tas/rgb/bridge_prompt_ms_tcn_gtea.py