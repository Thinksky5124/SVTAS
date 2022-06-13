###
 # @Author       : Thyssen Wen
 # @Date         : 2022-06-13 16:04:40
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-06-13 16:05:11
 # @Description  : Test script
 # @FilePath     : /ETESVS/script/test.sh
### 

export CUDA_VISIBLE_DEVICES=1
python main.py  --test -c config/gtea/feature/sliding_conformer_split1.yaml --weights=output/Sliding_ConFormer_gtea_split1/Sliding_ConFormer_gtea_split1_best.pkl