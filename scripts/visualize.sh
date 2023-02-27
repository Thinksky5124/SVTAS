###
 # @Author       : Thyssen Wen
 # @Date         : 2022-07-17 10:38:57
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-02-26 20:51:36
 # @Description  : Visualization Script
 # @FilePath     : /SVTAS/scripts/visualize.sh
### 

# visualize label
# gtea
python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 64
# 50salads
# python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 256
# # breakfast
# python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/breakfast/mapping_fine.txt output/results/imgs --sliding_windows 128

# # visualize cam image
# python tools/visualize/cam_visualization.py -c config/cam_visualize/mobinetv2tsm_3Dtcn_visualize.yaml -o output --method gradcam++
# python tools/visualize/cam_visualization.py -c config/cam_visualize/asformer_visualize.py -o output --method gradcam
# python tools/visualize/cam_visualization.py -c config/cam_visualize/segformer_visualize.py -o output --method gradcam