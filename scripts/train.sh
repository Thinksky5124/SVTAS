###
 # @Author       : Thyssen Wen
 # @Date         : 2022-05-22 17:05:58
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-03-02 16:06:45
 # @Description  : train script
 # @FilePath     : /SVTAS/scripts/train.sh
### 
export CUDA_VISIBLE_DEVICES=1

# mstcn 1538574472
# asformer 19980125
### gtea ###
# python tools/launch.py --mode train --validate -c config/svtas/rgb/swin_transformer_3d_small_taseformer_gtea.py --seed 400
python tools/launch.py --mode train --validate -c config/svtas/feature/block_recurrent_transformer.py --seed 400
# python tools/launch.py --mode train --validate -c config/svtas/rgb/swin_transformer_3d_small_asformer_50salads.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/asrf_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/ms_tcn_gtea.py --seed 0
# python tools/launch.py --mode train --validate -c config/svtas/feature/ms_tcn_gtea.py --seed 0
