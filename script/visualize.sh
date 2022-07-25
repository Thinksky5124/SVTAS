###
 # @Author       : Thyssen Wen
 # @Date         : 2022-07-17 10:38:57
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2022-07-19 09:55:07
 # @Description  : Visualization Script
 # @FilePath     : /ETESVS/script/visualize.sh
### 

# gtea
# python tools/convert_pred2img.py output/results/pred_gt_list data/gtea/mapping.txt output/results/imgs --sliding_windows 128
# 50salads
# python tools/convert_pred2img.py output/results/pred_gt_list data/50salads/mapping.txt output/results/imgs --sliding_windows 600
# breakfast
python tools/convert_pred2img.py output/results/pred_gt_list data/breakfast/mapping_fine.txt output/results/imgs --sliding_windows 128