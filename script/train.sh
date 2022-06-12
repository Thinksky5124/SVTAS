###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-06-12 20:29:18
 # @Description  : train script
 # @FilePath     : /ETESVS/script/train.sh
### 
python main.py --validate -c config/50salads/rgb/multi_modality/segmentation_clip_split1.yaml --seed 0

# python main.py --validate -c config/gtea/rgb/action_segmentation/mobinetv2tsm_mstcn_split1.yaml --seed 0

# python main.py --validate -c config/gtea/rgb/multi_modality/segmentation_clip_split1.yaml --seed 0 -o resume_epoch=21