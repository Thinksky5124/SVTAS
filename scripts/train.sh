###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-11-23 15:56:11
 # @Description  : train script
 # @FilePath     : /SVTAS/script/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

# mstcn 1538574472
# asformer 19980125
### gtea ###
# python tools/launch.py --mode train --validate -c config/svtas/rgb/mobilenetv2_50salads.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/efficientformer_50salads.py --seed 0
python tools/launch.py --mode train --validate -c config/svtas/rgb/efficientformer_breakfast.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/efficientnet_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/mobilenetv3_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/visual_transformer_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/efficientnet_gtea.py --seed 0 --use_tensorboard
