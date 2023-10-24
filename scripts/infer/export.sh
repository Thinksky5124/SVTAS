###
 # @Author       : Thyssen Wen
 # @Date         : 2022-09-24 20:54:46
 # @LastEditors  : Thyssen Wen
 # @LastEditTime : 2023-10-23 20:22:36
 # @Description  : Infer script
 # @FilePath     : /SVTAS/scripts/infer/export.sh
### 

# export model tensorRT
# python tools/launch.py --mode export -c config/export/swinv2_export_tensorRT.py
# export model TVM
python tools/launch.py --mode export -c config/export/swinv2_export_tvm.py