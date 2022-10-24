###
 # @Author       : Thyssen Wen
 # @Date         : 2022-07-17 10:38:57
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-10-24 12:03:25
 # @Description  : Visualization Script
 # @FilePath     : /SVTAS/script/visualize.sh
### 

# visualize label
# gtea
python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 128
# 50salads
python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
# breakfast
python tools/visualize/convert_pred2img.py output/results/pred_gt_list data/breakfast/mapping_fine.txt output/results/imgs --sliding_windows 128

# visualize cam image
python tools/visualize/cam_visualization.py -c config/cam_visualize/mobinetv2tsm_3Dtcn_visualize.yaml -o output --method gradcam++