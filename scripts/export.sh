###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-22 15:09:34
 # @Description  : Infer script
 # @FilePath     : /SVTAS/scripts/export.sh
### 

# export model
python tools/launch.py --mode export -c config/export/swinv2_export_tensorRT.py