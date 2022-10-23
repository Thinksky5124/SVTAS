###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-22 21:33:10
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python main.py --mode train --validate -c config/gtea/slvit_split1.yaml