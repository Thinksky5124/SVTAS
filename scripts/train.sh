###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-02-23 15:45:15
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

# mstcn 1538574472
# asformer 19980125
### gtea ###
python tools/launch.py --mode train --validate -c config/svtas/rgb/swin_transformer_3d_tiny_asformer_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/rgb/swin_transformer_3d_small_asformer_50salads.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/asrf_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/ms_tcn_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/ms_tcn_gtea.py --seed 0
