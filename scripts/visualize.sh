###
 # @Author       : Thyssen Wen
 # @Date         : 2022-07-17 10:38:57
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-17 11:00:20
 # @Description  : Visualization Script
 # @FilePath     : /SVTAS/scripts/visualize.sh
### 
export CUDA_VISIBLE_DEVICES=1
# visualize label
# gtea
# python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 64
# 50salads
# python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 256
# # breakfast
# python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/breakfast/mapping.txt output/results/imgs --sliding_windows 128

# # visualize cam image
python tools/launch.py --mode visualize -c config/cam_visualize/dynamic_swin_v2_transformer_fc_visualize.py

# # visualize loss landspace
# python tools/visualize/visualize_loss.py -c config/svtas/feature/asformer_gtea.py -w output/Stream_ASFormer_64x2_gtea_split4/Stream_ASFormer_64x2_gtea_split4_best.pt -o ./output/visulize_loss