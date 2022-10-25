###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-25 16:43:57
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

### gtea ###
python main.py --mode train --validate -c config/tas/ms_tcn/ms_tcn_gtea_split1.py