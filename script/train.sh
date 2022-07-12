###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-12 09:50:01
 # @Description  : train script
 # @FilePath     : /ETESVS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=0
# python main.py --validate -c config/gtea/rgb/action_segmentation/mobinetv2tsm_mstcn_split1.yaml --seed 0
python main.py --validate -c config/gtea/rgb/img_recognition/vit_asformer_split1.yaml --seed 0

# python main.py --validate -c config/gtea/rgb/action_segmentation/timesformer_conformer_split1.yaml --seed 0

# python main.py --validate -c config/gtea/rgb/multi_modality/segmentation_clip_split1.yaml --seed 0 -o resume_epoch=21