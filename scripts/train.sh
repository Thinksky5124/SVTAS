###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-02-07 15:32:57
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=0

# mstcn 1538574472
# asformer 19980125
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/rgb/swin_transformer_3d_asformer_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/tas/feature/asformer_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/tas/feature/tasegformer_50salads.py --seed 19980125
# python tools/launch.py --mode train --validate -c config/tas/feature/tasegformer_50salads.py --seed 19990924
# python tools/launch.py --mode train --validate -c config/tas/feature/segformer_50salads.py --seed 19980108
