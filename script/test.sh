###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-06-29 14:46:46
 # @Description  : Test script
 # @FilePath     : /ETESVS/script/test.sh
### 

export CUDA_VISIBLE_DEVICES=1
python main.py  --test -c config/gtea/rgb/multi_modality/bridge_prompt_split1.yaml --weights=output/Bridge_Prompt_gtea_split1/Bridge_Prompt_gtea_split1_best.pkl