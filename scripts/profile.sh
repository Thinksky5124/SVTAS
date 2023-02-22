###
 # @Author       : Thyssen Wen
 # @Date         : 2023-02-22 15:29:28
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-02-22 15:30:34
 # @Description  : file content
 # @FilePath     : /SVTAS/scripts/profile.sh
### 
export CUDA_VISIBLE_DEVICES=0

### gtea ###
python tools/launch.py --mode profile -c config/svtas/rgb/swin_transformer_3d_fc_clip_50salads.py -w output/SwinTransformer3D_FC_64x4_50salads_split1/SwinTransformer3D_FC_64x4_50salads_split1_best.pt